import numpy as np
import os
from Dataset.config import *


def load_raw_file(filename, shape):
    return np.fromfile(filename, dtype=np.uint8).reshape(shape)


def extract_patch(vol, z, y, x, patch_size):
    return vol[z: z + patch_size, y: y + patch_size, x: x + patch_size]


def create_directory(data_dirs):
    images_dir = data_dirs["train"]
    masks_dir = data_dirs["test"]
    folders = data_dirs["folders"]
    for folder in folders:
        os.makedirs(os.path.join(images_dir, folder), exist_ok=True)
        os.makedirs(os.path.join(masks_dir, folder), exist_ok=True)


def save_patch(img, seg, patch_size, num_patches, rock_type):
    rng = np.random.RandomState(metadata[rock_type]["seed"])
    num = metadata[rock_type]["number"]
    zmax, ymax, xmax = img.shape
    for i in range(num_patches):
        z = int(rng.randint(0, zmax - patch_size + 1))
        y = int(rng.randint(0, ymax - patch_size + 1))
        x = int(rng.randint(0, xmax - patch_size + 1))
        patch_img = extract_patch(img, z, y, x, patch_size).squeeze()
        patch_seg = extract_patch(seg, z, y, x, patch_size).squeeze()
        fname = f'patch_{num:02d}_i_{i:05d}_z_{z:03d}_y_{y:03d}_x_{x:03d}.npy'
        if i + 1 <= int(num_patches * 0.8):
            path = os.path.join(data_directory["train"], data_directory["folders"][0])
            np.save(os.path.join(path, fname), patch_img)
            path = os.path.join(data_directory["train"], data_directory["folders"][1])
            np.save(os.path.join(path, fname), patch_seg)
        else:
            path = os.path.join(data_directory["test"], data_directory["folders"][0])
            np.save(os.path.join(path, fname), patch_img)
            path = os.path.join(data_directory["test"], data_directory["folders"][1])
            np.save(os.path.join(path, fname), patch_seg)


def main():
    num_patches = 500
    patch_size = 128
    width, height, depth = dimensions["width"], dimensions["height"], dimensions["depth"]
    create_directory(data_directory)
    for rock_type in metadata:
        img = load_raw_file(metadata[rock_type]["original"], (depth, height, width))
        seg = load_raw_file(metadata[rock_type]["segmented"], (depth, height, width))
        save_patch(img, seg, patch_size, num_patches, rock_type)


if __name__ == '__main__':
    main()
