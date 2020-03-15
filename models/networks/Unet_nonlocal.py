from .U_Net import *
from models.layers.non_local_layer import NONLocalBlockND
from tensorflow.python.keras import utils

def unet_nonlocal(input_height, input_width,dimension=2,
                  nonlocal_mode='embedded_gaussian', nonlocal_sf=4,bn_layer=True):
    inputs = layers.Input(shape=(input_height, input_width, 6))

    #downsampling
    encoder0 = conv_block(inputs, 32)
    nonlocal0 = NONLocalBlockND(encoder0, encoder0.shape.as_list()[-1] // 4, dimension=dimension, mode=nonlocal_mode,
                                sub_sample_factor=nonlocal_sf, bn_layer=bn_layer)
    encoder_pool0 = layers.MaxPool2D(2, strides=2)(nonlocal0)

    encoder1 = conv_block(encoder_pool0, 64)
    encoder_pool1=layers.MaxPool2D(2,strides=2)(encoder1)

    encoder2 = conv_block(encoder_pool1, 128)
    encoder_pool2=layers.MaxPool2D(2,strides=2)(encoder2)#(nonlocal2)

    encoder_pool3, encoder3 = encoder_block(encoder_pool2, 256)
    center = conv_block(encoder_pool3, 512)
    #upsampling
    decoder3 = decoder_block(center, encoder3, 256)
    decoder2 = decoder_block(decoder3, encoder2, 128)
    decoder1 = decoder_block(decoder2, encoder1, 64)
    decoder0 = decoder_block(decoder1, encoder0, 32)

    # with Deep Surpervision
    dsv3 = dsv_block(decoder3, 1, 8)
    dsv2 = dsv_block(decoder2, 1, 4)
    dsv1 = dsv_block(decoder1, 1, 2)
    dsv0 = dsv_block(decoder0, 1, 1)
    outputs=layers.concatenate([dsv0,dsv1,dsv2,dsv3],axis=-1)
    outputs = layers.Conv2D(1, kernel_size=1, activation="sigmoid", kernel_initializer=initializer)(outputs)

    # without Deep Supervision
    #outputs = layers.Conv2D(1, kernel_size=1, activation="sigmoid", kernel_initializer=initializer)(decoder0)
    model = models.Model(inputs=[inputs], outputs=[outputs])
    print('Compiling Model.')
    model.compile(loss=dice_loss, optimizer=adam,metrics=[dice,prec,recall,hd,assd])
    return model


def xunet_nonlocal(input_height, input_width,dimension=2,
                  nonlocal_mode='embedded_gaussian', nonlocal_sf=4,bn_layer=True):
    inputs = layers.Input(shape=(input_height, input_width, 6))

    #downsampling
    encoder0 = xconv_block(inputs, 32)
    nonlocal0 = NONLocalBlockND(encoder0, encoder0.shape.as_list()[-1] // 4, dimension=dimension, mode=nonlocal_mode,
                                sub_sample_factor=nonlocal_sf, bn_layer=bn_layer)
    encoder_pool0 = layers.MaxPool2D(2, strides=2)(nonlocal0)


    encoder1 = xconv_block(encoder_pool0, 64)
    nonlocal1 = NONLocalBlockND(encoder1,encoder1.shape.as_list()[-1]//4,dimension=dimension,mode=nonlocal_mode,
                              sub_sample_factor=nonlocal_sf,bn_layer=bn_layer)
    encoder_pool1 = layers.MaxPool2D(2,strides=2)(nonlocal1)

    encoder2 = xconv_block(encoder_pool1, 128)
    nonlocal2=NONLocalBlockND(encoder2,encoder2.shape.as_list()[-1]//4,dimension=dimension,mode=nonlocal_mode,
                              sub_sample_factor=nonlocal_sf,bn_layer=bn_layer)
    encoder_pool2=layers.MaxPool2D(2,strides=2)(nonlocal2)


    encoder3 = xconv_block(encoder_pool2, 256)
    nonlocal3 = NONLocalBlockND(encoder3, encoder3.shape.as_list()[-1] // 4, dimension=dimension, mode=nonlocal_mode,
                                sub_sample_factor=nonlocal_sf, bn_layer=bn_layer)
    encoder_pool3 = layers.MaxPool2D(2,strides=2)(nonlocal3)

    #encoder_pool1, encoder1 = xencoder_block(encoder_pool0, 64)
    #encoder_pool2, encoder2 = xencoder_block(encoder_pool1, 128)
    #encoder_pool3, encoder3 = xencoder_block(encoder_pool2, 256)

    center = xconv_block(encoder_pool3, 512)
    nonlocal4 = NONLocalBlockND(center, center.shape.as_list()[-1] // 4, dimension=dimension, mode=nonlocal_mode,
                                sub_sample_factor=nonlocal_sf, bn_layer=bn_layer)
    #upsampling
    decoder3 = xdecoder_block(nonlocal4, encoder3, 256)
    decoder2 = xdecoder_block(decoder3, encoder2, 128)
    decoder1 = xdecoder_block(decoder2, encoder1, 64)
    decoder0 = xdecoder_block(decoder1, encoder0, 32)

    # with Deep Surpervision
    dsv3 = dsv_block(decoder3, 1, 8)
    dsv2 = dsv_block(decoder2, 1, 4)
    dsv1 = dsv_block(decoder1, 1, 2)
    dsv0 = dsv_block(decoder0, 1, 1)
    outputs=layers.concatenate([dsv0,dsv1,dsv2,dsv3],axis=-1)
    outputs = layers.Conv2D(1, kernel_size=1, activation="sigmoid", kernel_initializer=initializer)(outputs)

    # without Deep Supervision
    #outputs = layers.Conv2D(1, kernel_size=1, activation="sigmoid", kernel_initializer=initializer)(decoder0)
    model = models.Model(inputs=[inputs], outputs=[outputs])
    print('Compiling Model.')
    model.compile(loss=dice_loss, optimizer=adam,metrics=[dice,prec,recall,hd,assd])
    return model

