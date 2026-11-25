import os

import numpy as np
import tensorflow as tf
import random
import time
from Model.UNet3D import build_model
from Loss_function import loss_total
from utils import *
import glob
from Load_Data import *


random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)

num = 100
number = 'all'
path_img_train = './Dataset/dataset/train/images'
path_seg_train = './Dataset/dataset/train/masks'
img_files_train, seg_files_train = load_path(path_img_train, path_seg_train, num=num, number=number)

index = np.random.permutation(len(img_files_test))
batches = [index[i: i + 2] for i in range(0, len(index), 2)]
x_test, y_test = load_data([img_files_test[j] for j in batches[0]], [seg_files_test[j] for j in batches[0]])

input_shape = (128, 128, 128, 1)
model = build_model(input_shape, filter_num=8, prob=0.5)
model.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)

path = 'checkpoints/' + str(num) + '/' + number + '/0.5/'
checkpoint_path = path + 'model'
ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=15)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

bce = tf.keras.losses.BinaryCrossentropy()

@tf.function
def train_step(xt, yt):
    with tf.GradientTape(persistent=True) as tape:
        yp = model(xt, training=True)
        yp = tf.squeeze(yp)
        loss = bce(yt, yp)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    m1, m2, m3 = dice_coefficient(yt, yp), precision_smooth(yt, yp), recall_smooth(yt, yp)
    return loss, m1, m2, m3

EPOCHS = 10
batch_size = 2
train_loss = []
val_loss = []
for epoch in range(EPOCHS):
    start = time.time()
    step_loss = []
    epoch_time = []
    k = 0
    index = np.random.permutation(len(img_files_train))
    batches = [index[i: i + batch_size] for i in range(0, len(index), batch_size)]
    for i in batches:
        x, y = load_data([img_files_train[j] for j in i], [seg_files_train[j] for j in i])
        loss, metric1, metric2, metric3 = train_step(x, y)
        step_loss.append((loss.numpy(), metric1.numpy(), metric2.numpy(), metric3.numpy()))
        print('Step {:03d} | ET {:.2f} min, Loss:{:.6f}, DC:{:.6f}, PS:{:.6f}, RS:{:.6f}'.format(k + 1, (time.time() - start) / 60, *list(step_loss[-1])), end='\n')
        k = k + 1
    train_loss.append(np.mean(step_loss, axis=0))
    if (epoch + 1) % 2 == 0:
        save_path = ckpt_manager.save()
        print(f"Checkpoint Saved: {save_path}")
    if optimizer.lr >= 1e-6:
        optimizer.lr.assign(optimizer.lr * 0.5)
np.savetxt(path + 'train_loss.txt', train_loss)
np.savetxt(path + 'val_loss.txt', val_loss)
