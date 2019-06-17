# pylint: disable=C0103
"""
Keras model for Deepwatershed Detection
"""

from keras.layers.convolutional import Conv2D
from keras.layers import Input
from keras.models import Model
from sheet_id.models.FCN import FCN

def DWD(input_shape=(500, 500, 1), n_classes=124, n_energybins=10):
    """
    Create a deepwatershed model. The model takes an image of size input_shape
    (default to (500,500,1)). The model predicts 3 types of outputs:
    	- energy_output: quantized energy on every pixel
        - class_output: class prediction on every pixel
        - bbox_output: bounding box dimension on every pixel
    """

    model = FCN(input_shape=input_shape, n_classes=n_classes)
    input_map = Input(shape=input_shape)
    output_featuremaps = model(input_map)
    energy_output = Conv2D(n_energybins, (1, 1), activation='relu',
                           padding='same', name='energy_map')(output_featuremaps)
    class_output = Conv2D(n_classes, (1, 1), activation='relu',
                          padding='same', name='class_map')(output_featuremaps)
    bbox_output = Conv2D(2, (1, 1), activation='relu',
                         padding='same', name='bbox_map')(output_featuremaps)
    dwd_model = Model(inputs=[input_map], outputs=[energy_output, class_output, bbox_output])
    return dwd_model
