from typing import Tuple
import numpy as np
import torch
from torch.autograd import Function
import torch.nn as nn

try:
    import pointops_cuda
except ImportError:
    import warnings
    import os
    from torch.utils.cpp_extension import load

    warnings.warn("Unable to load pointops_cuda cpp extension.")
    pointops_cuda_src = os.path.join(os.path.dirname(__file__), "../src")
    pointops_cuda = load('pointops_cuda', [
        pointops_cuda_src + '/pointops_api.cpp',
        pointops_cuda_src + '/ballquery/ballquery_cuda.cpp',
        pointops_cuda_src + '/ballquery/ballquery_cuda_kernel.cu',
        pointops_cuda_src + '/knnquery/knnquery_cuda.cpp',
        pointops_cuda_src + '/knnquery/knnquery_cuda_kernel.cu',
        pointops_cuda_src + '/knnquery_heap/knnquery_heap_cuda.cpp',
        pointops_cuda_src + '/knnquery_heap/knnquery_heap_cuda_kernel.cu',
        pointops_cuda_src + '/grouping/grouping_cuda.cpp',
        pointops_cuda_src + '/grouping/grouping_cuda_kernel.cu',
        pointops_cuda_src + '/grouping_int/grouping_int_cuda.cpp',
        pointops_cuda_src + '/grouping_int/grouping_int_cuda_kernel.cu',
        pointops_cuda_src + '/interpolation/interpolation_cuda.cpp',
        pointops_cuda_src + '/interpolation/interpolation_cuda_kernel.cu',
        pointops_cuda_src + '/sampling/sampling_cuda.cpp',
        pointops_cuda_src + '/sampling/sampling_cuda_kernel.cu'
    ], build_directory=pointops_cuda_src, verbose=False)


class FurthestSampling(Function):
    @staticmethod
    def forward(ctx, xyz, m):
        """
        input: xyz: (b, n, 3) and n > m, m: int32
        output: idx: (b, m)
        """
        assert xyz.is_contiguous()
        b, n, _ = xyz.size()
        idx = torch.cuda.IntTensor(b, m)
        temp = torch.cuda.FloatTensor(b, n).fill_(1e10)
        pointops_cuda.furthestsampling_cuda(b, n, m, xyz, temp, idx)
        return idx

    @staticmethod
    def backward(xyz, a=None):
        return None, None


furthestsampling = FurthestSampling.apply


class Gathering(Function):
    @staticmethod
    def forward(ctx, features, idx):
        """
        input: features: (b, c, n), idx : (b, m) tensor
        output: (b, c, m)
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()
        b, c, n = features.size()
        m = idx.size(1)
        output = torch.cuda.FloatTensor(b, c, m)
        pointops_cuda.gathering_forward_cuda(b, c, n, m, features, idx, output)
        ctx.for_backwards = (idx, c, n)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        idx, c, n = ctx.for_backwards
        b, m = idx.size()
        grad_features = torch.cuda.FloatTensor(b, c, n).zero_()
        grad_out_data = grad_out.data.contiguous()
        pointops_cuda.gathering_backward_cuda(b, c, n, m, grad_out_data, idx, grad_features.data)
        return grad_features, None


gathering = Gathering.apply


class NearestNeighbor(Function):
    @staticmethod
    def forward(ctx, unknown: torch.Tensor, known: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find the three nearest neighbors of unknown in known
        input: unknown: (b, n, 3), known: (b, m, 3)
        output: dist2: (b, n, 3) l2 distance to the three nearest neighbors
                idx: (b, n, 3) index of 3 nearest neighbors
        """
        assert unknown.is_contiguous()
        assert known.is_contiguous()
        b, n, _ = unknown.size()
        m = known.size(1)
        dist2 = torch.cuda.FloatTensor(b, n, 3)
        idx = torch.cuda.IntTensor(b, n, 3)
        pointops_cuda.nearestneighbor_cuda(b, n, m, unknown, known, dist2, idx)
        return torch.sqrt(dist2), idx

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None


nearestneighbor = NearestNeighbor.apply


class Interpolation(Function):
    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        Performs weight linear interpolation on 3 features
        input: features: (b, c, m) features descriptors to be interpolated from
               idx: (b, n, 3) three nearest neighbors of the target features in features
               weight: (b, n, 3) weights
        output: (b, c, n) tensor of the interpolated features
        """
        features = features.contiguous()
        assert features.is_contiguous()
        assert idx.is_contiguous()
        assert weight.is_contiguous()
        b, c, m = features.size()
        n = idx.size(1)
        ctx.interpolation_for_backward = (idx, weight, m)
        output = torch.cuda.FloatTensor(b, c, n)
        pointops_cuda.interpolation_forward_cuda(b, c, m, n, features, idx, weight, output)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        input: grad_out: (b, c, n)
        output: grad_features: (b, c, m), None, None
        """
        idx, weight, m = ctx.interpolation_for_backward
        b, c, n = grad_out.size()
        grad_features = torch.cuda.FloatTensor(b, c, m).zero_()
        grad_out_data = grad_out.data.contiguous()
        pointops_cuda.interpolation_backward_cuda(b, c, n, m, grad_out_data, idx, weight, grad_features.data)
        return grad_features, None, None


interpolation = Interpolation.apply


class Grouping(Function):
    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        input: features: (b, c, n), idx : (b, m, nsample) containing the indicies of features to group with
        output: (b, c, m, nsample)
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()
        b, c, n = features.size()
        _, m, nsample = idx.size()
        output = torch.tensor((b, c, m, nsample), dtype=torch.float32, device='cuda')
        pointops_cuda.grouping_forward_cuda(b, c, n, m, nsample, features, idx, output)
        ctx.for_backwards = (idx, n)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        input: grad_out: (b, c, m, nsample)
        output: (b, c, n), None
        """
        idx, n = ctx.for_backwards
        b, c, m, nsample = grad_out.size()
        grad_features = torch.zeros((b, c, n), dtype=torch.int32, device='cuda')
        grad_out_data = grad_out.data.contiguous()
        pointops_cuda.grouping_backward_cuda(b, c, n, m, nsample, grad_out_data, idx, grad_features.data)
        return grad_features, None


grouping = Grouping.apply


class GroupingInt(Function):
    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        input: features: (b, c, n), idx : (b, m, nsample) containing the indicies of features to group with
        output: (b, c, m, nsample)
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()
        b, c, n = features.size()
        _, m, nsample = idx.size()
        output = torch.cuda.LongTensor(b, c, m, nsample)
        pointops_cuda.grouping_int_forward_cuda(b, c, n, m, nsample, features, idx, output)
        return output

    @staticmethod
    def backward(ctx, a=None):
        return None, None


grouping_int = GroupingInt.apply


class BallQuery(Function):
    @staticmethod
    def forward(ctx, radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
        """
        input: radius: float, radius of the balls
               nsample: int, maximum number of features in the balls
               xyz: torch.Tensor, (b, n, 3) xyz coordinates of the features
               new_xyz: torch.Tensor, (b, m, 3) centers of the ball query
        output: (b, m, nsample) tensor with the indicies of the features that form the query balls
        """
        assert xyz.is_contiguous()
        assert new_xyz.is_contiguous()
        b, n, _ = xyz.size()
        m = new_xyz.size(1)
        idx = torch.zeros((b, m, nsample), dtype=torch.int32, device='cuda')
        pointops_cuda.ballquery_cuda(b, n, m, radius, nsample, new_xyz, xyz, idx)
        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


ballquery = BallQuery.apply


def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    import numpy as np
    return torch.clamp(dist, 0.0, np.inf)


class KNNQueryNaive(Function):
    @staticmethod
    def forward(ctx, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor = None) -> Tuple[torch.Tensor]:
        """
        KNN Indexing
        input: nsample: int32, Number of neighbor
               xyz: (b, n, 3) coordinates of the features
               new_xyz: (b, m, 3) centriods
            output: idx: (b, m, nsample)
        """
        if new_xyz is None:
            new_xyz = xyz
        b, m, _ = new_xyz.size()
        n = xyz.size(1)

        '''
        idx = torch.zeros(b, m, nsample).int().cuda()
        for i in range(b):
            dist = pairwise_distances(new_xyz[i, :, :], xyz[i, :, :])
            [_, idxs] = torch.sort(dist, dim=1)
            idx[i, :, :] = idxs[:, 0:nsample]
        '''

        # '''
        # new_xyz_repeat = new_xyz.repeat(1, 1, n).view(b, m * n, 3)
        # xyz_repeat = xyz.repeat(1, m, 1).view(b, m * n, 3)
        # dist = (new_xyz_repeat - xyz_repeat).pow(2).sum(dim=2).view(b, m, n)
        dist = (new_xyz.repeat(1, 1, n).view(b, m * n, 3) - xyz.repeat(1, m, 1).view(b, m * n, 3)).pow(2).sum(
            dim=2).view(b, m, n)
        [_, idxs] = torch.sort(dist, dim=2)
        idx = idxs[:, :, 0:nsample].int()
        # '''
        return idx

    @staticmethod
    def backward(ctx):
        return None, None, None


knnquery_naive = KNNQueryNaive.apply


class KNNQuery(Function):
    @staticmethod
    def forward(ctx, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor = None) -> Tuple[torch.Tensor]:
        """
        KNN Indexing
        input: nsample: int32, Number of neighbor
               xyz: (b, n, 3) coordinates of the features
               new_xyz: (b, m, 3) centriods
            output: idx: (b, m, nsample)
                   ( dist2: (b, m, nsample) )
        """
        if new_xyz is None:
            new_xyz = xyz
        xyz = xyz.contiguous()
        new_xyz = new_xyz.contiguous()
        assert xyz.is_contiguous()
        assert new_xyz.is_contiguous()
        b, m, _ = new_xyz.size()
        n = xyz.size(1)
        idx = torch.zeros((b, m, nsample), dtype=torch.int32, device='cuda')
        dist2 = torch.zeros((b, m, nsample), dtype=torch.float32, device='cuda')
        pointops_cuda.knnquery_cuda(b, n, m, nsample, xyz, new_xyz, idx, dist2)
        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None


knnquery = KNNQuery.apply


class KNNQuery_Heap(Function):
    @staticmethod
    def forward(ctx, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor = None) -> Tuple[torch.Tensor]:
        """
        KNN Indexing
        input: nsample: int32, Number of neighbor
               xyz: (b, n, 3) coordinates of the features
               new_xyz: (b, m, 3) centriods
            output: idx: (b, m, nsample)
                   ( dist2: (b, m, nsample) )
        """
        if new_xyz is None:
            new_xyz = xyz
        assert xyz.is_contiguous()
        assert new_xyz.is_contiguous()
        b, m, _ = new_xyz.size()
        n = xyz.size(1)
        idx = torch.cuda.IntTensor(b, m, nsample).zero_()
        dist2 = torch.cuda.FloatTensor(b, m, nsample).zero_()
        pointops_cuda.knnquery_heap_cuda(b, n, m, nsample, xyz, new_xyz, idx, dist2)
        ctx.mark_non_differentiable(idx)
        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None


knnquery_heap = KNNQuery_Heap.apply


class QueryAndGroup(nn.Module):
    """
    Groups with a ball query of radius
    parameters:
        radius: float32, Radius of ball
        nsample: int32, Maximum number of features to gather in the ball
    """

    def __init__(self, radius=None, nsample=32, use_xyz=True, return_idx=False):
        super(QueryAndGroup, self).__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz
        self.return_idx = return_idx

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor = None, features: torch.Tensor = None,
                idx: torch.Tensor = None) -> torch.Tensor:
        """
        input: xyz: (b, n, 3) coordinates of the features
               new_xyz: (b, m, 3) centriods
               features: (b, c, n)
               idx: idx of neighbors
               # idxs: (b, n)
        output: new_features: (b, c+3, m, nsample)
              #  grouped_idxs: (b, m, nsample)
        """
        if new_xyz is None:
            new_xyz = xyz
        if idx is None:
            if self.radius is not None:
                idx = ballquery(self.radius, self.nsample, xyz, new_xyz)
            else:
                # idx = knnquery_naive(self.nsample, xyz, new_xyz)   # (b, m, nsample)
                # idx = knnquery(self.nsample, xyz, new_xyz)  # (b, m, nsample)
                idx = knnquery_heap(self.nsample, xyz, new_xyz)  # (b, m, nsample)
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping(xyz_trans, idx)  # (b, 3, m, nsample)
        # grouped_idxs = grouping(idxs.unsqueeze(1).float(), idx).squeeze(1).int()  # (b, m, nsample)
        grouped_xyz_diff = grouped_xyz - new_xyz.transpose(1, 2).unsqueeze(-1)
        if features is not None:
            grouped_features = grouping(features, idx)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz_diff, grouped_features], dim=1)  # (b, 3+c, m, nsample)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz_diff

        if self.return_idx:
            return new_features, grouped_xyz, idx.long()
            # (b,c,m,k), (b,3,m,k), (b,m,k)
        else:
            return new_features, grouped_xyz
