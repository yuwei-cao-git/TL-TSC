# import glob
import random
import numpy as np
import torch
import torchvision.transforms.v2 as transforms

def normalize_point_cloud(xyz):
    # Center and scale spatial coordinates
    centroid = np.mean(xyz, axis=0)
    xyz_centered = xyz - centroid
    max_distance = np.max(np.linalg.norm(xyz_centered, axis=1))
    xyz_normalized = xyz_centered / (max_distance + 1e-8)

    return xyz_normalized


def center_point_cloud(xyz):
    xyz_min = np.amin(xyz, axis=0, keepdims=True)
    xyz_max = np.amax(xyz, axis=0, keepdims=True)
    xyz_center = (xyz_min + xyz_max) / 2
    xyz_center[0][-1] = xyz_min[0][-1]
    xyz = xyz - xyz_center
    return xyz

def angle_axis(angle, axis):
    # type: (float, np.ndarray) -> float
    r"""Returns a 4x4 rotation matrix that performs a rotation around axis by angle

    Parameters
    ----------
    angle : float
        Angle to rotate by
    axis: np.ndarray
        Axis to rotate about

    Returns
    -------
    torch.Tensor
        3x3 rotation matrix
    """
    u = axis / np.linalg.norm(axis)
    cosval, sinval = np.cos(angle), np.sin(angle)

    # yapf: disable
    cross_prod_mat = np.array([[0.0, -u[2], u[1]],
                                [u[2], 0.0, -u[0]],
                                [-u[1], u[0], 0.0]])

    R = torch.from_numpy(
        cosval * np.eye(3)
        + sinval * cross_prod_mat
        + (1.0 - cosval) * np.outer(u, u)
    )
    # yapf: enable
    return R.float()


def rotate_points(coords, x=None):
    rotation_angle = np.random.uniform() * 2 * np.pi
    axis = np.array([0.0, 1.0, 0.0])
    # Rotate point cloud
    rot_mat = angle_axis(rotation_angle, axis)

    aug_coords = coords
    aug_coords[:, :3] = np.matmul(aug_coords[:, :3], rot_mat.t())
    if x is None:
        aug_x = None
    else:
        aug_x = x
        aug_x[:, :3] = np.matmul(aug_x[:, :3], rot_mat.t())

    return aug_coords, aug_x


def point_rotate_perturbation(coords, angle_sigma=0.06, angle_clip=0.18, x=None):
    angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
    Rx = angle_axis(angles[0], np.array([1.0, 0.0, 0.0]))
    Ry = angle_axis(angles[1], np.array([0.0, 1.0, 0.0]))
    Rz = angle_axis(angles[2], np.array([0.0, 0.0, 1.0]))

    rot_mat = torch.matmul(torch.matmul(Rz, Ry), Rx)
    aug_coords = coords
    aug_coords[:, :3] = np.matmul(aug_coords[:, :3], rot_mat.t())
    if x is None:
        aug_x = None
    else:
        aug_x = x
        aug_x[:, :3] = np.matmul(aug_x[:, :3], rot_mat.t())

    return aug_coords, aug_x


def point_removal(coords, n, x=None):
    # Get list of ids
    idx = list(range(np.shape(coords)[0]))
    random.shuffle(idx)  # shuffle ids
    idx = np.random.choice(
        idx, n, replace=False
    )  # pick points randomly removing up to 10% of points

    # Remove random values
    aug_coords = coords[idx, :]  # remove coords
    if x is None:  # remove x
        aug_x = None
    else:
        aug_x = x[idx, :]

    return aug_coords, aug_x


def point_translate(coords, translate_range=0.1, x=None):
    translation = np.random.uniform(-translate_range, translate_range)
    coords[:, 0:3] += translation
    if x is None:  # remove x
        aug_x = None
    else:
        aug_x = x + translation
    return coords, aug_x


def point_jitter(coords, std=0.01, clip=0.05, x=None):
    # Generate jittered noise with a normal distribution
    jittered_data = np.clip(
        np.random.normal(loc=0.0, scale=std, size=(coords.shape[0], 3)), -clip, clip
    )
    # Apply jitter to the coordinates
    coords[:, 0:3] += jittered_data

    # Conditionally apply jitter to x if it is provided
    if x is not None:
        x[:, 0:3] += jittered_data

    return coords, x


def random_scale(coords, lo=0.9, hi=1.1, x=None):
    scaler = np.random.uniform(lo, hi)
    aug_coords = coords * scaler
    if x is None:
        aug_x = None
    else:
        aug_x = x * scaler
    return aug_coords, aug_x


def random_noise(coords, n, dim=1, x=None):
    # Random standard deviation value
    random_noise_sd = np.random.uniform(0.01, 0.025)

    # Add/Subtract noise
    if np.random.uniform(0, 1) >= 0.5:  # 50% chance to add
        aug_coords = coords + np.random.normal(
            0, random_noise_sd, size=(np.shape(coords)[0], 3)
        )
        if x is None:
            aug_x = None
        else:
            aug_x = x + np.random.normal(
                0, random_noise_sd, size=(np.shape(x)[0], dim)
            )  # added [0] and dim
    else:  # 50% chance to subtract
        aug_coords = coords - np.random.normal(
            0, random_noise_sd, size=(np.shape(coords)[0], 3)
        )
        if x is None:
            aug_x = None
        else:
            aug_x = x - np.random.normal(
                0, random_noise_sd, size=(np.shape(x)[0], dim)
            )  # added [0] and dim

    # Randomly choose up to 10% of augmented noise points
    use_idx = np.random.choice(
        aug_coords.shape[0],
        n,
        replace=False,
    )
    aug_coords = aug_coords[use_idx, :]  # get random points
    aug_coords = np.append(coords, aug_coords, axis=0)  # add points
    if x is None:
        aug_x = None
    else:
        aug_x = aug_x[use_idx, :]  # get random point values
        aug_x = np.append(x, aug_x, axis=0)  # add random point values # ADDED axis=0

    return aug_coords, aug_x


def pointCloudTransform(xyz, pc_feat, target, rot=False):
    # Point Removal
    n = random.randint(round(len(xyz) * 0.9), len(xyz))
    aug_xyz, aug_feats = point_removal(xyz, n, x=pc_feat)
    aug_xyz, aug_feats = random_noise(aug_xyz, n=(len(xyz) - n), x=aug_feats)
    #aug_xyz, aug_feats = random_scale(xyz, x=pc_feat)
    aug_xyz, aug_feats = point_translate(aug_xyz, x=aug_feats)
    # aug_xyz, aug_feats = point_jitter(aug_xyz, x=aug_feats)
    if rot:
        aug_xyz, aug_feats = point_rotate_perturbation(aug_xyz, x=aug_feats)
        aug_xyz, aug_feats = rotate_points(aug_xyz, x=aug_feats)

    target = target

    return aug_xyz, aug_feats, target

    
def image_augment(img, image_transform, tile_size):
    if image_transform == "random":
        transform = transforms.RandomApply(
            torch.nn.ModuleList(
                [
                    transforms.RandomCrop(size=(tile_size, tile_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
                    transforms.RandomRotation(degrees=(0, 180)),
                    transforms.RandomAffine(
                        degrees=(30, 70),
                        translate=(0.1, 0.3),
                        scale=(0.5, 0.75),
                    ),
                ]
            ),
            p=0.3,
        )
    elif image_transform == "compose":
        transform = transforms.Compose(
            [
                # transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=interpol_mode),  # 3 is bicubic
                transforms.RandomCrop(size=(tile_size, tile_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
                transforms.RandomRotation(degrees=(0, 180)),
                transforms.RandomAffine(
                    degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)
                ),
                #transforms.Resize(224, 224)
            ]
        )
    else:
        transform = None

    if transform is None:
        aug_image = None
    else:
        aug_image = transform(img)

    return aug_image
