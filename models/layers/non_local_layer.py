from tensorflow.python.keras import layers,backend,models
import tensorflow as tf
from utils.config import *

def NONLocalBlockND(x, inter_channels=None, dimension=2, mode='embedded_gaussian',
                 sub_sample_factor=4, bn_layer=True):
    assert dimension in [1, 2, 3]

    # print('Dimension: %d, mode: %s' % (dimension, mode))
    sub_sample_factor_list = sub_sample_factor if isinstance(sub_sample_factor, list) else [sub_sample_factor]
    in_channels = x.shape.as_list()[-1]
    if inter_channels is None:
        inter_channels =in_channels // 2
        if inter_channels == 0:
            inter_channels = 1

    if dimension == 3:
        conv_nd = layers.Conv3D
        max_pool = layers.MaxPool3D
    elif dimension == 2:
        conv_nd = layers.Conv2D
        max_pool = layers.MaxPool2D
    else:
        conv_nd = layers.Conv1D
        max_pool = layers.MaxPool1D
    bn = layers.BatchNormalization

    x_size = x.shape.as_list()
    batch_size = x_size[0]  # (?，48，48，16，64)

    # g=>(b, t, h, w, c)->(b,t/4, h/4, w/4, 0.25c)->(b, thw/64, 0.25c)
    # phi  =>(b, c, t, h, w)[->(b, thw/64, 0.25c)]
    g_x=conv_nd(filters=inter_channels,kernel_size=1, strides=1,kernel_initializer=initializer, padding="same")(x)
    phi_x=conv_nd(filters=inter_channels,kernel_size=1, strides=1,kernel_initializer=initializer, padding="same")(x)
    if any(ss > 1 for ss in sub_sample_factor_list):
        g_x= max_pool(pool_size=sub_sample_factor, strides=sub_sample_factor)(g_x)
        phi_x = max_pool(pool_size=sub_sample_factor, strides=sub_sample_factor)(phi_x)
    g_size = g_x.shape.as_list()
    size = 1
    for i in range(dimension):
        size = g_size[i + 1] * size
    g_x = tf.reshape(g_x, [-1, int(size), inter_channels])  # (?，?，16)
    #g_x = backend.permute_dimensions(g_x, (0, 2, 1))  # (b, 0.25c, thw/64)(?，16，?)
    phi_size = phi_x.shape.as_list()
    size = 1
    for i in range(dimension):
        size = phi_size[i + 1] * size
    phi_x = tf.reshape(phi_x, [-1, int(size), inter_channels])  # (?，?，16)
    phi_x = backend.permute_dimensions(phi_x,(0, 2, 1))# modified non-local (b, 0.25c, thw/64)(?，16，?)
    # theta=>(b, c, t, h, w)[->(b, t, h, w, 0.25c)]->(b,0.25c,thw)
    # f=>(b, thw, 0.5c)dot(b, 0.5c, twh) = (b, thw, thw)
    theta_x=conv_nd(filters=inter_channels,kernel_size=1, strides=1,kernel_initializer=initializer, padding="same")(x)
    theta_size = theta_x.shape.as_list()
    size = 1
    for i in range(dimension):
        size = theta_size[i + 1] * size
    theta_x = tf.reshape(theta_x, [-1, int(size), inter_channels])  # (?，?，16)
    #theta_x = backend.permute_dimensions(theta_x, (0, 2, 1))  # (?，16，?)
    f = tf.matmul(theta_x,phi_x)  # (?，?，?) (b, thw, thw/64)
    f_div_C = layers.Activation("softmax")(f)
    # f_div_C = F.softmax(f, dim=-1)

    # (b, thw, thw/64) dot (b, thw/64, c/4)= (b, thw, c/4)->(b, t, h, w, c/4)->(b, t, h, w, c)
    y = tf.matmul(f_div_C, g_x)  # (?，16，?)
    #y = backend.permute_dimensions(y, (0, 2, 1))  # (?，?，16)
    # y = y.permute(0, 2, 1).contiguous()
    y = tf.reshape(y, [-1, *x_size[1:(dimension + 1)], inter_channels])  # (?，48，48，16，16)
    W_y=conv_nd(filters=in_channels, kernel_size=1, strides=1, kernel_initializer=initializer, padding="same")(y) # (?，48，48，16，64)
    if bn_layer:
        W_y=bn()(W_y)
    z = W_y + x

    return z



