import random

import numpy as np
import scipy
import tensorflow as tf
import glob
import os
import time
from utils import predict

from Model.UNet3D import build_model
from Load_Data import load_data

random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)

num = 150
number = 'all'

path_img_test = './Dataset/dataset/test/images'
path_seg_test = './Dataset/dataset/test/masks'
img_files_test = sorted(glob.glob(os.path.join(path_img_test, '*.npy')))
seg_files_test = sorted(glob.glob(os.path.join(path_seg_test, '*.npy')))


input_shape = (128, 128, 128, 1)
model = build_model(input_shape, filter_num=8, prob=0.4)
model.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)

path = 'checkpoints/' + str(num) + '/' + number + '/0.4/'
checkpoint_path = path + "model"
ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=15)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

os.makedirs(os.path.join(path, "pred"), exist_ok=True)
os.makedirs(os.path.join(path, "au"), exist_ok=True)
os.makedirs(os.path.join(path, "eu"), exist_ok=True)

num = len(img_files_test)
T = 20
start = time.time()
dc_list = []
io_list = []
for i in range(num):
    x, y = load_data([img_files_test[i]], [seg_files_test[i]])
    pred, au, eu, scores = predict(model, x, y, time=T)
    pd_list.append(pred)
    au_list.append(au)
    eu_list.append(eu)
    dc_list.append(np.mean(scores[0]))
    io_list.append(np.mean(scores[1]))
    print('Step {:03d} | ET {:.2f} min'.format(i + 1, (time.time() - start) / 60), end='\n')

end = time.time()

np.savetxt(path + 'dice.txt', dc_list)
np.savetxt(path + 'iou.txt', io_list)


