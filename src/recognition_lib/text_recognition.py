"""
肤色识别模型
"""
import os
import cv2
import numpy as np
from keras.applications.xception import Xception
from keras.applications import xception
from keras.models import Model
from keras.layers import *
from keras.models import *
from keras.callbacks import *
from keras import backend as K 
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.utils.generic_utils import get_custom_objects

def softmax(x):
    ndim = K.ndim(x)
    if ndim == 1:
        raise ValueError('Cannot apply softmax to a tensor that is 1D')
    elif ndim == 2:
        import tensorflow as tf
        return tf.nn.softmax(x)
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D. '
                         'Received input: %s' % x)
get_custom_objects().update({'mySoftmax': Activation(softmax)})



class textRecognition():
    """
    文本识别模型
    """
    def get_session(self):
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        return tf.Session(config=tf_config)

    def __init__(self, modelPath=''):
        self.modelPath = modelPath
        label_dict = {'0': 'text', '1': 'norm'}
        with tf.variable_scope('text'):
            self.input_tensor = Input(shape=(299, 299, 3))
            self.base_model = Xception(input_tensor=self.input_tensor,
                                  weights=None, include_top=False, pooling='avg')
            self.x = self.base_model.output
            self.feature = BatchNormalization()(self.x, training=False)
            self.classify = Dense(2, activation='mySoftmax',
                             use_bias=False, name='text')(self.feature)
            self.model = Model(self.input_tensor, self.classify)
            self.model.load_weights(self.modelPath)
            self.model.predict(np.zeros((1, 299,299,3)))

    def read_img(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (299, 299))
        img = (np.array(img) - 127.5) / 127.5
        return img

    def recognition(self, img):
        img = self.read_img(img)
        imgs = np.reshape(img, (-1, 299, 299, 3))
        predict = self.model.predict(imgs)
        predict_result = np.max(predict, axis=-1)[0]
        predict_index = str(np.argmax(predict, axis=-1)[0])
        return predict_index