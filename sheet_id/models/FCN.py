from keras.models import Model
from keras.layers import Input, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose, MaxPooling2D
from keras.layers.merge import concatenate

from sheet_id.utils.BilinearUpSampling import BilinearUpSampling2D

def FCN(input_shape=(500, 500, 1), n_classes=124):
    """
    Return an FCN architecture used for DeepScores semantic segmentation
    """
    input_tensor = Input(shape=input_shape)

    # Encoder
    conv2 = Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu', name='conv2')(input_tensor)
    pool2 = MaxPooling2D(pool_size=(2,2), padding='same', name='pool2')(conv2)
    dropout2 = Dropout(rate=0.15, name='dropout2')(pool2)

    conv3 = Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu', name='conv3')(dropout2)
    pool3 = MaxPooling2D(pool_size=(2,2), padding='same', name='pool3')(conv3)
    dropout3 = Dropout(rate=0.15, name='dropout3')(pool3)

    conv4 = Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu', name='conv4')(dropout3)
    pool4 = MaxPooling2D(pool_size=(2,2), padding='same', name='pool4')(conv4)
    dropout4 = Dropout(rate=0.15, name='dropout4')(pool4)

    conv5 = Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu', name='conv5')(dropout4)
    pool5 = MaxPooling2D(pool_size=(2,2), padding='same', name='pool5')(conv5)
    dropout5 = Dropout(rate=0.15, name='dropout5')(pool5)

    conv6 = Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu', name='conv6')(dropout5)
    pool6 = MaxPooling2D(pool_size=(2,2), padding='same', name='pool6')(conv6)
    dropout6 = Dropout(rate=0.15, name='dropout6')(pool6)

    conv7 = Conv2D(filters=4096, kernel_size=(3,3), padding='same', name='conv7')(dropout6)

    # Decoder
    conv_t1 = Conv2DTranspose(filters=512, kernel_size=(4,4), strides=(2,2), padding='same', name='conv_t1')(conv7)
    conv_t1_up = BilinearUpSampling2D(target_size=tuple(pool5.get_shape().as_list()[1:3]))(conv_t1)

    stacked_1 = concatenate(inputs=[conv_t1_up, pool5], axis=-1, name='stacked_1')
    fuse_1_1 = Conv2D(filters=512, kernel_size=(1,1), activation='relu', padding='same', name='fuse_1_1')(stacked_1)
    fuse_1_2 = Conv2D(filters=512, kernel_size=(1,1), activation='relu', padding='same', name='fuse_1_2')(fuse_1_1)

    conv_t2 = Conv2DTranspose(filters=256, kernel_size=(4,4), strides=(2,2), padding='same', name='conv_t2')(fuse_1_2)
    conv_t2_up = BilinearUpSampling2D(target_size=tuple(pool4.get_shape().as_list()[1:3]))(conv_t2)

    stacked_2 = concatenate(inputs=[conv_t2_up, pool4], axis=-1, name='stacked_2')
    fuse_2_1 = Conv2D(filters=256, kernel_size=(1,1), activation='relu', padding='same', name='fuse_2_1')(stacked_2)
    fuse_2_2 = Conv2D(filters=256, kernel_size=(1,1), activation='relu', padding='same', name='fuse_2_2')(fuse_2_1)

    conv_t3 = Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2), padding='same', name='conv_t3')(fuse_2_2)
    conv_t3_up = BilinearUpSampling2D(target_size=tuple(pool3.get_shape().as_list()[1:3]))(conv_t3)

    stacked_3 = concatenate(inputs=[conv_t3_up, pool3], axis=-1, name='stacked_3')
    fuse_3_1 = Conv2D(filters=128, kernel_size=(1,1), activation='relu', padding='same', name='fuse_3_1')(stacked_3)
    fuse_3_2 = Conv2D(filters=128, kernel_size=(1,1), activation='relu', padding='same', name='fuse_3_2')(fuse_3_1)

    conv_t4 = Conv2DTranspose(filters=64, kernel_size=(4,4), strides=(2,2), padding='same', name='conv_t4')(fuse_3_2)
    conv_t4_up = BilinearUpSampling2D(target_size=tuple(pool2.get_shape().as_list()[1:3]))(conv_t4)

    stacked_4 = concatenate(inputs=[conv_t4_up, pool2], axis=-1, name='stacked_4')
    fuse_4_1 = Conv2D(filters=64, kernel_size=(1,1), activation='relu', padding='same', name='fuse_4_1')(stacked_4)
    fuse_4_2 = Conv2D(filters=64, kernel_size=(1,1), activation='relu', padding='same', name='fuse_4_2')(fuse_4_1)

    # Final upscaling
    deconv_final = Conv2DTranspose(filters=n_classes, kernel_size=(16,16), strides=(2,2),
                                   padding='same', name='deconv_final')(fuse_4_2)
    output = BilinearUpSampling2D(target_size=tuple(input_tensor.get_shape().as_list()[1:3]), name='output')(deconv_final)

    return Model(inputs=[input_tensor], outputs=[output])
