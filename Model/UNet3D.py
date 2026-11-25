import tensorflow as tf
import random

from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, ZeroPadding3D, Concatenate, Resizing, SpatialDropout3D, UpSampling3D, Cropping3D
from tensorflow.keras.models import Model
from tensorflow.keras.activations import softplus

random.seed(0)
tf.random.set_seed(0)


def build_model(input_shape, filter_num=4, prob=0.1):
    inputs = Input(shape=input_shape)

    c0 = inputs

    c1 = Conv3D(filter_num * 2, 3, padding='same', activation='relu', kernel_initializer='he_normal')(c0)
    c1 = Conv3D(filter_num * 2, 3, padding='same', activation='relu', kernel_initializer='he_normal')(c1)
    p1 = MaxPooling3D(pool_size=(2, 2, 2))(c1)
    p1 = SpatialDropout3D(prob)(p1)

    c2 = Conv3D(filter_num * 4, 3, padding='same', activation='relu', kernel_initializer='he_normal')(p1)
    c2 = Conv3D(filter_num * 4, 3, padding='same', activation='relu', kernel_initializer='he_normal')(c2)
    p2 = MaxPooling3D(pool_size=(2, 2, 2))(c2)
    p2 = SpatialDropout3D(prob)(p2)

    c3 = Conv3D(filter_num * 8, 3, padding='same', activation='relu', kernel_initializer='he_normal')(p2)
    c3 = Conv3D(filter_num * 8, 3, padding='same', activation='relu', kernel_initializer='he_normal')(c3)
    p3 = MaxPooling3D(pool_size=(2, 2, 2))(c3)
    p3 = SpatialDropout3D(prob)(p3)

    c4 = Conv3D(filter_num * 16, 3, padding='same', activation='relu', kernel_initializer='he_normal')(p3)
    c4 = Conv3D(filter_num * 16, 3, padding='same', activation='relu', kernel_initializer='he_normal')(c4)
    p4 = MaxPooling3D(pool_size=(2, 2, 2))(c4)
    p4 = SpatialDropout3D(prob)(p4)

    c5 = Conv3D(filter_num * 32, 3, padding='same', activation='relu', kernel_initializer='he_normal')(p4)
    c5 = Conv3D(filter_num * 32, 3, padding='same', activation='relu', kernel_initializer='he_normal')(c5)

    u5 = UpSampling3D(size=(2, 2, 2))(c5)
    u5 = Concatenate()([u5, c4])
    c6 = Conv3D(filter_num * 16, 3, padding='same', activation='relu', kernel_initializer='he_normal')(u5)
    c6 = Conv3D(filter_num * 16, 3, padding='same', activation='relu', kernel_initializer='he_normal')(c6)

    u6 = UpSampling3D(size=(2, 2, 2))(c6)
    u6 = Concatenate()([u6, c3])
    c7 = Conv3D(filter_num * 8, 3, padding='same', activation='relu', kernel_initializer='he_normal')(u6)
    c7 = Conv3D(filter_num * 8, 3, padding='same', activation='relu', kernel_initializer='he_normal')(c7)

    u7 = UpSampling3D(size=(2, 2, 2))(c7)
    u7 = Concatenate()([u7, c2])
    c8 = Conv3D(filter_num * 4, 3, padding='same', activation='relu', kernel_initializer='he_normal')(u7)
    c8 = Conv3D(filter_num * 4, 3, padding='same', activation='relu', kernel_initializer='he_normal')(c8)

    u8 = UpSampling3D(size=(2, 2, 2))(c8)
    u8 = Concatenate()([u8, c1])
    c9 = Conv3D(filter_num * 2, 3, padding='same', activation='relu', kernel_initializer='he_normal')(u8)
    c9 = Conv3D(filter_num * 2, 3, padding='same', activation='relu', kernel_initializer='he_normal')(c9)

    outputs = Conv3D(1, 1, padding='same', activation='sigmoid')(c9)

    model = Model(inputs, outputs)

    return model