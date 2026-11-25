import numpy as np
import os
import glob


def load_path(path_img_train, path_seg_train, num=400, number='all'):
    path_img = sorted(os.listdir(path_img_train))
    path_seg = sorted(os.listdir(path_seg_train))
    img_files_train, seg_files_train = [], []
    if number == 'all':
        for i, j in zip(path_img, path_seg):
            img_files_train.extend(sorted(glob.glob(os.path.join(os.path.join(path_img_train, i), '*.npy')))[0:num])
            seg_files_train.extend(sorted(glob.glob(os.path.join(os.path.join(path_seg_train, j), '*.npy')))[0:num])
    else:
        for i, j in zip(path_img, path_seg):
            if i != number and j != number:
                img_files_train.extend(sorted(glob.glob(os.path.join(os.path.join(path_img_train, i), '*.npy')))[0:num])
                seg_files_train.extend(sorted(glob.glob(os.path.join(os.path.join(path_seg_train, j), '*.npy')))[0:num])
    return sorted(img_files_train), sorted(seg_files_train)


def load_data(img_files, seg_files):
    imgs, segs = [], []
    for i, j in zip(img_files, seg_files):
        img = np.load(i)
        seg = np.load(j)
        imgs.append(img)
        segs.append(seg)
    imgs = np.array(imgs)
    segs = np.array(segs)
    return imgs.astype(np.float32) / 255.0, segs.astype(int)
