from .utils import *
from utils.config import *

def unet(input_height, input_width):
    inputs = layers.Input(shape=(input_height, input_width, 6))

    encoder_pool0, encoder0 = encoder_block(inputs, 32)
    encoder_pool1, encoder1 = encoder_block(encoder_pool0, 64)
    encoder_pool2, encoder2 = encoder_block(encoder_pool1, 128)
    encoder_pool3, encoder3 = encoder_block(encoder_pool2, 256)
    center = conv_block(encoder_pool3, 512)
    decoder3 = decoder_block(center, encoder3, 256)
    decoder2 = decoder_block(decoder3, encoder2, 128)
    decoder1 = decoder_block(decoder2, encoder1, 64)
    decoder0 = decoder_block(decoder1, encoder0, 32)
    # with Deep Surpervision
    dsv3 = dsv_block(decoder3, 1, 8)
    dsv2 = dsv_block(decoder2, 1, 4)
    dsv1 = dsv_block(decoder1, 1, 2)
    dsv0 = dsv_block(decoder0, 1, 1)
    outputs = layers.concatenate([dsv0, dsv1, dsv2, dsv3], axis=-1)

    # without Supervision
    #outputs = layers.Conv2D(1, (1, 1), activation="sigmoid",kernel_initializer=initializer)(decoder0)#(outputs)#
    model = models.Model(inputs=[inputs], outputs=[outputs])
    print('Compiling Model.')
    model.compile(loss=dice_loss, optimizer=adam,metrics=[dice,prec,recall,hd,assd])
    return model


