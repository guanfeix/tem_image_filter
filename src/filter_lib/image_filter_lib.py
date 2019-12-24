'''
服装图像过滤器lib:
    为服装打上相关TAG:
        1.图像面积是否达标
        2.是否为灰度图
        3.分辨率是否较高
        4.是否为暗系图
        5.是否为黑白图 
        6.人脸:
            1.人脸数量,人脸位置
            2.人脸肤色
        7.服装:
            1. 服装占比
            2. 服装种类
        8. 图像中是否存在大量文本
'''

import cv2
import numpy as np
from ..recognition_lib.complexion import complexionRecognition
from ..detect_all_lib.face_detection import faceDetect, alignment
from ..detect_all_lib.yolo_detection import YOLO
from ..recognition_lib.text_recognition import textRecognition


class ImageFilter(object):
    """
    服装图像过滤器,用于获取图像的各项信息
    """

    def __init__(self, face_detect_model_path='',
                 clothes_detect_model_path='',
                 complexion_model_path='',
                 text_model_path='',
                 resolution_threshold=260,
                 brightness_threshold=78,
                 min_side_threshold=720,
                 max_side_threshold=10000,
                 face_attributes_list=['complexion']):

        # 获取检测模型路径并加载人脸特征,服装检测模型
        self.face_detect_model_path = face_detect_model_path
        self.clothes_detect_model_path = clothes_detect_model_path
        self.complexion_model_path = complexion_model_path

        # 人脸模型初始化
        self.face_detect_model = faceDetect(self.face_detect_model_path)
        self.face_complexion_model = complexionRecognition(
            self.complexion_model_path)
        
        # 服装检测模型初始化
        self.clothes_detecter= YOLO(self.clothes_detect_model_path)
       
        # 分辨率最低阈值 亮度阈值 最小边阈值 最大边阈值
        self.resolution_threshold = resolution_threshold
        self.brightness_threshold = brightness_threshold
        self.min_side_threshold = min_side_threshold
        self.max_side_threshold = max_side_threshold
        
        # 人脸属性
        self.face_attributes_list = face_attributes_list

        # 获取文字识别模型地址，并加载文字识别模型
        self.text_model_path = text_model_path
        self.text_model = textRecognition(modelPath = self.text_model_path)
        
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
            return np.mean(img[:, :, 0]) == np.mean(img[:, :, 1])
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
            return img_resolution > self.resolution_threshold, img_resolution, self.resolution_threshold
        else:
            return None, None, self.resolution_threshold
        
    def cal_img_brightness(self, img):
        """
        使用HLS通道对lightness去除极值(亮度为0和255)后进行平均计算
        输入参数:
            img : cv2读取到的图像
        返回结果:
           dark:暗系图像
           norm:正常图像
        """
        if not img is None:
            if len(img.shape) == 2:
                return None, 'is_gray'
            else:
                hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
                lightness_result_ = hls_img[:, :, 1]
                lightness_result_ = lightness_result_[np.where((lightness_result_!=0)& (lightness_result_!=255))] # 删除极值
                lightness_result = np.sum(lightness_result_)/(hls_img.shape[0]*hls_img.shape[1])
                if lightness_result < self.brightness_threshold:
                    result = ('dark', 'filter-brightness-dark')
                else:
                    result = ('norm', '')
                return result
        else:
            return None, 'no image'

   

    def get_face_positions(self, img):
        """
        进行人脸检测，输出人脸数量和人脸位置
        输入参数：
            img: cv2读取的图像
        返回结果:
            faceNum: 人脸数量
            face_positions: 人脸位置列表
            landmark_alignment: 人脸角点
        """
        bounding_boxes, landmarks = self.face_detect_model.detect(img)
        faceNum = bounding_boxes.shape[0]
        if faceNum == 0:
            return faceNum, None, None
        else:
            face_positions = []
            img_h, img_w = img.shape[:2]
            for index, bounding_box in enumerate(bounding_boxes):
                xmin, ymin, xmax, ymax, _ = bounding_box
                face_positions.append({"xmin": round(xmin/img_w, 3), "ymin": round(
                    ymin/img_h, 3), "xmax": round(xmax/img_w, 3), "ymax": round(ymax/img_h, 3)})
            return faceNum, face_positions, landmarks

    def get_face_complexion(self, img, face_num, landmarks):
        """
        进行人脸检测,并输出数量,人脸性别年龄肤色还在测试ing,下一版本更新
        输入参数:
            img: 原始图像
            face_num: 人脸数量
            landmarks: 人脸角点
        返回结果:
            face_complexion_labels : 人脸肤色列表
                black: 黑人
                not-black: 非黑人
        """
        complexion_label_dict = {"0": "not_black", "1": "black"}
        if not img is None and face_num > 0:
            imgs, face_complexion_labels = [], []
            for face_index in range(face_num):
                landmark_alignment = []
                for i in range(5):
                    landmark_alignment.append(
                        [int(landmarks[i][face_index]), int(landmarks[i + 5][face_index])])
                alignment_img = alignment(img, landmark_alignment)
                imgs.append(alignment_img)
            predictions_complexion = self.face_complexion_model.recognition(imgs)
            for index in range(len(imgs)):
                complexion_label_index = predictions_complexion[index]
                face_complexion_labels.append(
                    complexion_label_dict[complexion_label_index])
            return face_complexion_labels
        else:
            return None

    def get_face_attributes(self, img, face_arributes_name):
        """
        用于获取人脸属性,传入参数:
            img: 传入图像
            face_arributes_name: 需要识别的人脸属性名称，支持的属性列表为:['complexion']
        """

        if not face_arributes_name in self.face_attributes_list:
            print('[ERROR] face_arributes_name is {}, not in predict face_attributes_list: {}'.format(
                face_arributes_name, self.face_attributes_list))
        else:
            faceNum, face_positions, landmarks = self.get_face_positions(img)
            if face_arributes_name == self.face_attributes_list[0]:
                # 改动
                return face_positions, self.get_face_complexion(img, faceNum, landmarks)


    def get_clothes_category_positions(self, img):
        """
        进行服装检测
        输入参数:
            img: 检测图像        
        """
        if not img is None:
            return self.clothes_detecter.detect_image(img)
        else:
            return None
          
    def get_img_text(self, img):
        """
        判断图像内是否存在大量文本
        返回参数:
            'text': 图像中存在大量文本
            'norm': 图像为正常图像
        """
        text_label_dict = {"0": "text", "1": "norm"}
        if img is not None:
            if len(img.shape) == 2:
                return 'is_grey'
            else:
                # result_label = self.text_model.recognition(img)
                # print(result_label)
                # result_index = text_label_dict[result_label]
                return self.text_model.recognition(img)
