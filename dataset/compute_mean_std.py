import cv2
import os
import numpy as np
from tqdm import tqdm

RESOLUTION = 96 # Ideally we shouldn't be resizing but I'm lacking memory

if __name__ == "__main__":
    data = []
    path = "/mnt/g/ovf/ovf_tl_dataset/tile_128/train"
    files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".npz")]

    for file in tqdm(files, miniters=256):
        img = np.load(file, allow_pickle=True)
        data.append(img)

    data = np.array(data, np.float32) / 255 # Must use float32 at least otherwise we get over float16 limits
    print("Shape: ", data.shape)

    means = []
    stdevs = []
    for i in range(3):
        pixels = data[:,:,:,i].ravel()
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    print("means: {}".format(means))
    print("stdevs: {}".format(stdevs))
    print('transforms.Normalize(mean = {}, std = {})'.format(means, stdevs))