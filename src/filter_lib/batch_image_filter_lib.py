'''
服装图像过滤器lib:
    为服装打上相关TAG:
        1.图像面积是否达标
        2.分辨率是否较高
        3.是否为暗系图
        4.人脸:
            1.人脸数量,人脸位置
            2.人脸肤色
        5.服装:
            1. 服装占比
            2. 服装种类
        6. 图像中是否存在大量文本
'''

import os
import cv2
import numpy as np
import sys
from sklearn.externals import joblib
sys.path.append("..")
from recognition_lib.complexion import complexionRecognition
from detect_all_lib.face_detection import faceDetect, alignment
from detect_all_lib.yolo_detection import YOLO
from recognition_lib.text_recognition import textRecognition

class imageFilterGeneral():
    """
    图像通用过滤器,用于获取图像的通用信息
    """
    def __init__(self, resolution_threshold=260,
                 brightness_threshold = 78,
                 min_side_threshold = 720, 
                 max_side_threshold = 10000):

        # 图像分辨率阈值
        self.resolution_threshold = resolution_threshold
        # 图像明暗度阈值
        self.brightness_threshold = brightness_threshold
        # 图像最小边阈值
        self.min_side_threshold = min_side_threshold
        # 图像最大边阈值
        self.max_side_threshold = max_side_threshold

    def cal_img_min_side(self, img):
        """
        计算图像的最小边长
        输入参数:
            img: cv2读取到的图像
        返回结果:
            True: 图像最小边大于等于最小边阈值
            False: 图像最小边小于最小边阈值
        """
        if not img is None:
            img_h, img_w, _ = img.shape
            return  min(img_h, img_w) >= self.min_side_threshold
        else:
            return None
        
    def cal_img_max_side(self, img):
        """
        计算图像的最大边长
        输入参数:
            img: cv2读取到的图像
        返回结果:
            True: 图像最大边小于等于最大边阈值
            False: 图像最大边大于最大边阈值
        """
        if not img is None:
            img_h, img_w, _ = img.shape
            return  max(img_h, img_w) <= self.max_side_threshold
        else:
            return None

    def is_img_gray(self, img):
        """
        判断图像是否为灰度图像(如果图像不存在,则返回None):
        输入参数:
            img : cv2读取到的图像
        返回结果:
            True  : 图像为灰度图像
            False : 图像不是灰度图像
        """
        if not img is None:
            img_channel = img.shape
            if len(img_channel) == 2:
                return True
            return  np.mean(img[:, :, 0]) == np.mean(img[:, :, 1])
        else:
            return None

    def is_img_black(self, img):
        """"
        判断图像是否为黑白图像（如果图像不存在，则返回None):
        输入参数：
            img: cv2读取到的图像
        返回结果：
            True: 图像为黑白图像
            False: 图像不是黑白图像
            None: 图像不存在
        """
        if not img is None:
            img_h, img_w = img.shape[:2]
            img_bw = np.reshape(img, (-1, 3))
            black_num = len(np.where(np.sum(img_bw, axis=1) == 255*3)[0])
            white_num = len(np.where(np.sum(img_bw, axis=1) == 0)[0])
            return black_num+white_num == img_h*img_w
        else:
            return None

    def cal_img_resolution(self, img):
        """
        计算图像的分辨率,如果图像的分辨率高于阈值则返回True,小于阈值则返回False(如果图像不存在,则返回None)
        输入参数:
            img : cv2读取到的图像
        返回结果:
            True  : 分辨率高于阈值
            False : 分辨率低于阈值

        """
        if not img is None:
            img_resolution = cv2.Laplacian(img, cv2.CV_64F).var()
            return img_resolution > self.resolution_threshold
        else:
            return None
        
    def cal_img_brightness(self, img):
        """
        使用HLS通道对lightness去除极值(亮度为0和255)后进行平均计算
        输入参数:
            img : cv2读取到的图像
        返回结果:
           False:暗系图像
           True:正常图像
        """
        if not img is None:
            if len(img.shape) == 2:
                return None
            else:
                hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
                lightness_result_ = hls_img[:, :, 1]
                lightness_result_ = lightness_result_[np.where((lightness_result_!=0)& (lightness_result_!=255))] # 删除极值
                lightness_result = np.sum(lightness_result_)/(hls_img.shape[0]*hls_img.shape[1])
                return lightness_result >= self.brightness_threshold
        else:
            return None


class imageFilterModel():
    """
    服装图像过滤器,用于获取图像的各项信息
    """

    def __init__(self, face_detect_model_path='',
                 clothes_detect_model_path='',
                 complexion_model_path='',
                 text_model_path = ''):

        # 加载人脸检测模型地址，服装检测模型地址，肤色识别模型地址，文字识别模型地址
        self.face_detect_model_path = face_detect_model_path
        self.clothes_detect_model_path = clothes_detect_model_path
        self.complexion_model_path = complexion_model_path
        self.text_model_path = text_model_path

        # 人脸模型初始化
        self.face_detect_model = faceDetect(self.face_detect_model_path)
        self.face_complexion_model = complexionRecognition(
            self.complexion_model_path)
        
        # 服装检测模型初始化
        self.clothes_detecter= YOLO(self.clothes_detect_model_path)
        
        # 文字识别模型初始化
        self.text_model = textRecognition(modelPath = self.text_model_path)
        

    def get_face_positions(self, batch_img):
        """
        进行人脸检测，输出人脸数量和人脸位置
        输入参数：
            img: cv2读取的图像
        返回结果:
            bounding_boxes_list: 人脸位置列表
            landmarks_list: 人脸角点
        """
        bounding_boxes_list, landmarks_list = self.face_detect_model.detect_batch(batch_img)
        for index, bounding_boxs in enumerate(bounding_boxes_list):
            if len(bounding_boxs) != 0:
                face_positions = []
                image = batch_img[index]
                img_h, img_w = image.shape[:2]
                for bounding_box in bounding_boxs:
                    xmin, ymin, xmax, ymax, _ = bounding_box
                    face_positions.append([round(xmin/img_w, 3), round(ymin/img_h, 3), round(xmax/img_w, 3), round(ymax/img_h, 3)])
                bounding_boxes_list[index] = face_positions
        return bounding_boxes_list, landmarks_list

    def get_face_complexion(self, batch_img, landmarks_list):
        """
        进行人脸检测,并输出数量,人脸性别年龄肤色还在测试ing,下一版本更新
        输入参数:
            batch_img: 原始图像列表
            landmarks_list: 人脸角点列表
        返回结果:
            face_complexion_labels_results : 人脸肤色列表
                black: 黑人
                not-black: 非黑人
        """
        imageList = []
        complexion_label_dict = {"0": "not_black", "1": "black"}
        for index, img in enumerate(batch_img):
            landmarks = landmarks_list[index]
            imgs = []
            if not img is None and len(landmarks) != 0 and landmarks.shape[1] != 0:
                for face_index in range(landmarks.shape[1]):
                    landmark_alignment = []
                    for i in range(5):
                        landmark_alignment.append(
                            [int(landmarks[i][face_index]), int(landmarks[i + 5][face_index])])
                    alignment_img = alignment(img, landmark_alignment)
                    imgs.append(alignment_img)
            imageList.append(imgs)
        predictions_complexion_list  = self.face_complexion_model.recognition_batch(imageList)
        for index_, predictions_complexion in enumerate(predictions_complexion_list):
            if len(predictions_complexion) != 0:
                predictions_complexion_list[index_] = [complexion_label_dict[x] for x in predictions_complexion]
        return predictions_complexion_list

    def get_face_attributes(self, batch_img):
        """
        用于获取人脸属性,传入参数:
            batch_img: 传入图像列表
            返回: 
                face_positions_list: face 位置信息
                人脸肤色信息
        """
        face_positions_list, landmarks_list = self.get_face_positions(batch_img)

        return face_positions_list, self.get_face_complexion(batch_img, landmarks_list)


    def get_clothes_category_positions(self, batch_img):
        """
        进行服装检测
        输入参数:
            batch_img: 检测图像列表  
        返回参数:
            category_level_1_results = []   
        """
        category_level_1_results= []
        for image in batch_img:
            category_level_1_result = self.clothes_detecter.detect_image(image)
            category_level_1_results.append(category_level_1_result)
        return category_level_1_results
          
    def get_img_text(self, batch_img):
        """
        判断图像内是否存在大量文本
        返回参数:
            text_results = ['text', 'norm', 'norm', ...]
            'norm' : 图像为正常图像
            'text' : 图像中存在大量文本
        """
        text_results = self.text_model.recognition_batch(batch_img)
        return text_results
