import numpy as np
from tensorflow.keras import backend as K
import tensorflow as tf
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage.morphology import distance_transform_edt as edt


def dice_coefficient(y_true, y_pred):
    y_true_f = K.flatten(tf.cast(y_true,tf.float32))
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2 * intersection / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon()))


def iou_smooth(y_true, y_pred):
    y_true_f = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = K.cast(K.flatten(y_pred) > 0.5, tf.float32)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_pred_f) + K.sum(y_true_f) - intersection
    return (intersection / (union + K.epsilon()))


def predict(model, image, gt, time=10):
    p_hat = []
    for t in range(time):
        p_hat.append(model(image, training=True)[0])
    p_hat = np.array(p_hat)
    pred = np.mean(p_hat, axis=0)
    au = np.mean(p_hat * (1 - p_hat), axis=0)
    eu = np.var(p_hat, axis=0)  # eu = np.mean(p_hat ** 2, axis=0) - np.mean(p_hat, axis=0) ** 2
    dc = dice_coefficient(gt, pred)
    io = iou_smooth(gt, pred)
    return np.squeeze(pred), np.squeeze(au), np.squeeze(eu), (dc, io)





