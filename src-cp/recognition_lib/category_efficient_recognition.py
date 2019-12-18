"""efficient_net 品类二级标签识别"""
import cv2
import os
from keras.layers import *
import numpy as np
from keras.models import Model
from recognition_lib.efficient_net import EfficientNetB3
from PIL import Image
import tensorflow as tf
import keras.backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.backend.tensorflow_backend import set_session


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


class Efficient_Recognition():
    
    def get_session(self):
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        return tf.Session(config=tf_config)

    def __init__(self, not_predict_labels, modelPath='xxx.h5'):
        self.modelPath = modelPath
        if not os.path.exists(self.modelPath):
            print("{} is not exists".format(self.modelPath))
        self.IMAGE_SIZE = 300
        self.STYLE_NUM = 30
        self.not_predict_labels = not_predict_labels
        self.labels = ['T恤', '卫衣', '衬衫', 'polo衫', '打底-吊带-背心-抹胸', '泳装-内衣', '套衫', '大衣-风衣', '羽绒服-棉服', '夹克', '西服', '无袖外套-马甲', '披肩-斗篷', '连衣裙', '半身裙', '吊带连衣裙', '礼服',  '中长裤', '短裤', '连体裤', '内裤-泳裤', '针织套衫', '针织外套',  '皮毛短外套', '皮毛大衣', '皮夹克', '皮大衣', '牛仔长裤', '牛仔短裤', '牛仔外套']
        with tf.variable_scope('category_recognition'):
            input_tensor = Input(shape=(300,300,3))
            base_model = EfficientNetB3(input_tensor=input_tensor,weights=None,pooling='avg')
            x = base_model.layers[-3].output
            x = BatchNormalization()(x,training=False)
            classify3 = Dense(30, activation='mySoftmax',use_bias=False, name='classify3')(x)
            self.model = Model(input_tensor, classify3)
            self.model.load_weights(self.modelPath)
            self.model.predict(np.zeros((1, 300, 300, 3)))

    def preprocessImg(self, img):
        newImg = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (self.IMAGE_SIZE, self.IMAGE_SIZE))
        saveImg = (newImg - 127.5) / 127.5
        return saveImg

    def correctionLabel(self, level_one_label, predict, threshold):
        if level_one_label in self.not_predict_labels:
            return level_one_label, 1
        predict_precision = np.max(predict, axis=-1)
        if predict_precision > threshold:
            label_index = np.argmax(predict, axis=-1)
        else:
            labels_index_list = [self.labels.index(x) for x in self.level_match_dict[level_one_label]]
            predict_result = [predict[x] for x in labels_index_list]
            predict_max = np.max(predict_result, axis=-1)
            label_index = list(predict).index(predict_max)
        return self.labels[label_index], predict_precision


    def recognition(self, img,positions,min_precision=0.1):
        # 从原始图像中切割出需要进行识别的图像,进行预处理
        img_h, img_w, _ = img.shape
        img_list = []
        for position in positions:
            xmin, ymin, xmax, ymax,level_1_label = float(position['xmin']), float(position['ymin']), float(position['xmax']), float(position['ymax']),position['label']
            if level_1_label in self.not_predict_labels:
                continue
            xmin, ymin, xmax, ymax = int(xmin * img_w), int(ymin * img_h), int(xmax * img_w), int(ymax * img_h)
            cropImg = img[ymin:ymax, xmin:xmax]
            cropImg = self.preprocessImg(cropImg)
            img_list.append(cropImg)
        img_list = np.reshape(img_list, (-1, self.IMAGE_SIZE, self.IMAGE_SIZE, 3))

        # 开始进行识别
        predicts = self.model.predict(img_list)
        result = []
        for i in range(img_list.shape[0]):
            predict_result = predicts[i]
            predict_index = np.argmax(predict_result, axis=-1)
            predict_precision = np.max(predict_result, axis=-1)
            if not (predict_precision > min_precision):
                label = ''
            else:
                label = self.labels[predict_index]
            result.append([label,predict_precision])
        return result

