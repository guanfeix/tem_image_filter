"""
肤色识别模型
"""
import os
import numpy as np
from keras.applications.xception import Xception
from keras.applications import xception
from keras.models import Model
from keras.layers import *
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


class complexionRecognition():
    """
    肤色识别模型
    """
    def get_session(self):
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        return tf.Session(config=tf_config)
    
    def __init__(self, modelPath =''):
        self.modelPath = modelPath
        if not os.path.exists(self.modelPath):
            print("{} is not exists".format(self.modelPath))
        with tf.variable_scope('face_complexion'):
            self.base_model = Xception(input_shape=(112, 96, 3),weights = None,include_top=False,pooling='avg')
            self.input_tensor = Input((112,96, 3))
            self.x = Lambda(xception.preprocess_input)(self.input_tensor)
            self.x = self.base_model(self.x)
            self.x = Dense(2, activation='sigmoid', name='face_complexion')(self.x)
            self.model = Model(self.input_tensor, self.x)
            self.model.load_weights(self.modelPath) 
            self.model.predict(np.zeros((1,112,96,3)))
 
    def recognition(self,imgs):
        imgs = np.reshape(np.array(imgs)/255,(-1,112,96,3))
        predicts = self.model.predict(imgs)
        result = [] 
        for i in range(imgs.shape[0]):
            predict_result = predicts[i]
            predict_index = str(np.argmax(predict_result, axis=-1))
            result.append(predict_index)
        return result


     



        