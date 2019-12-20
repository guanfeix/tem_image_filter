import sys
import traceback
import logging

import numpy as np
from typing import List

from .detect_base import DetectBase

logger = logging.getLogger(__name__)


class QualityDetect(DetectBase):

    def __init__(self, filter, image_url, image_source='weibo',image_cv=None):

        super(QualityDetect, self).__init__(image_url, image_cv)
        self.filter = filter
        self.face_infos = None
        self.image_source = image_source
        self.uselessDetectClothesLabels = ['帽子', '鞋靴', '包']

    def get_image_resolution(self):
        clear, resolution = self.filter.cal_img_resolution(self.image_cv)
        return not clear, resolution

    def get_image_brightness(self):
        brightness, msg = self.filter.cal_img_brightness(self.image_cv)
        return brightness, msg

    def get_face_blur(self, pos):
        xmin, ymin, xmax, ymax = pos[0], pos[1], pos[2], pos[3]
        blur, resolution = self.filter.cal_img_resolution(
            self.image_cv[ymin:ymax, xmin:xmax])
        return not blur, resolution

    def get_face_attributes(self):

        # self.prepare_faceinfos()

        face_num = self.face_infos['face_num']
        landmarks = self.face_infos['face_landmarks']

        final_landmarks = []
        for index in range(face_num):
            final_landmarks.append(landmarks[:, index])

        logger.info('get face attr ...')
        complexion_list = self.filter.get_face_attributes(
            self.image_cv, final_landmarks, "complexion")
        logger.info('complexion ok')

        sex_list = self.filter.get_face_attributes(
            self.image_cv, final_landmarks, "sex")
        logger.info('sex ok')

        race_list = self.filter.get_face_attributes(
            self.image_cv, final_landmarks, "race")
        logger.info('race ok')

        return complexion_list, sex_list, race_list

    def get_image_faceinfo(self):
        # num, positions, landmarks = self.filter.get_img_face_info(self.image_cv)
        num, face_normal_positions, landmarks = self.filter.get_face_positions(self.image_cv)
        positions = []
        if face_normal_positions:
            positions = [list(i.values()) for i in face_normal_positions]
        logger.info('face number: {}, face_normal_positions: {}, positions: {}'.format(num, face_normal_positions, positions))
        return num, positions, landmarks, face_normal_positions

    def prepare_faceinfos(self):

        try:
            if not self.face_infos:
                face_num, face_positions, face_landmarks, face_normal_positions = self.get_image_faceinfo()
                face_positions = face_positions if face_num > 0 else []

                self.face_infos = {
                    'face_num': face_num,
                    'face_positions': face_positions,
                    'face_landmarks': face_landmarks,
                    'face_normal_positions': face_normal_positions
                }
                logger.info('face_infos %s', self.face_infos)
        except Exception as e:
            logger.error('prepare face info fail: {}'.format(e))
            logger.error(''.join(traceback.format_exception(*sys.exc_info())[-2:]))

    # def check_face_position(self, pos):
    #
    #     img_h, img_w, img_c = self.image_cv.shape
    #     xmin, ymin, xmax, ymax = pos[0], pos[1], pos[2], pos[3]
    #     face_w, face_h = xmax - xmin, ymax - ymin
    #
    #     is_big = face_h > 0.25 * img_h
    #     is_small = face_h < 0.05 * img_h or face_w < 40 or face_h < 40
    #     is_center = (xmin > 0.15 * img_w and xmin < 0.85 *
    #                  img_w) and (ymin > 0 and ymax < 0.85 * img_h)
    #     return is_big, is_small, is_center

    def judge_image_shape(self):
        img_h, img_w, _ = self.image_cv.shape
        if self.image_source == 'ins':
            return min(img_h, img_w) > 720
        elif self.image_source == 'weibo':
            return min(img_h, img_w) > 640 and max(img_h, img_w) < 1800

    def tag_base_info(self):

        base_tags = []

        normal = self.judge_image_shape()
        if not normal:
            base_tags.append('shape_unfit')

        blur, resolution = self.get_image_resolution()
        if blur:
            base_tags.append('filter-blur')

        brightness, msg = self.get_image_brightness()
        if brightness != 'norm':
            base_tags.append(msg)

        return base_tags, {'image-blur': blur, 'image-resolution': resolution, 'image-shape': self.image_cv.shape}

    def tag_faces_info(self, max_face_num=3):
        """
        逻辑先跑通后面在慢慢调整
        :param max_face_num:
        :return:
        """
        self.prepare_faceinfos()
        face_num = self.face_infos['face_num']

        if face_num == 0 or face_num > max_face_num:
            return True, ['no-face'], [], 0

        face_positions: List[List] = self.face_infos['face_positions']
        face_normal_positions = self.face_infos['face_normal_positions']
        # np array
        face_landmarks = self.face_infos['face_landmarks']

        face_filter_fail = True
        new_face_positions = []
        # img_w_f, img_h_f = float(img_w), float(img_h)
        # face_positions = [list(map(lambda x: round(x / img_h, 3), i)) for i in face_positions]
        logger.info('face_positions %s', face_positions)
        for index, pos in enumerate(face_positions):
            # 过滤高度小于0.05的人脸
            xmin, ymin, xmax, ymax = pos[0], pos[1], pos[2], pos[3]
            face_w, face_h = xmax - xmin, ymax - ymin

            if face_h > 0.05:
                new_face_positions.append(pos)
                if face_filter_fail and 0.4 > face_h:
                    face_filter_fail = False
                    logger.info('face_filter_fail: %s, pos: %s', face_filter_fail, pos)
            else:
                np.delete(face_landmarks, index, axis=1)


        new_face_num = len(new_face_positions)
        self.face_infos['face_landmarks'] = face_landmarks
        self.face_infos['face_num'] = new_face_num
        if face_filter_fail:
            msg = '人脸高度全部大于0.4或者人脸高度全部小于0.05'
            return False, msg, face_normal_positions, face_num
        if new_face_num > 3:
            face_filter_fail = True
            msg = 'face_num>3'
            return False, 'face_num>3', face_normal_positions, new_face_num
        elif new_face_num == 0:
            face_filter_fail = False
            msg = 'after filter no-face'
            return True, ['after filter no-face'], face_normal_positions, 0
        else:
            for index, pos in enumerate(new_face_positions):
                # 去除人脸高度小于0.05) 再次过滤face_num_1 大于3,face_h_1大于0.4

                xmin, ymin, xmax, ymax = pos[0], pos[1], pos[2], pos[3]
                face_w, face_h = xmax - xmin, ymax - ymin
                if face_h > 0.4:
                    msg = 'face_h_1>0.4 '
                    return False, [msg], [], new_face_num

        return True, None, face_normal_positions, new_face_num

    def tag_text_info(self):
        return self.filter.get_img_text(self.image_cv) != 'norm'

    def get_face_complexion(self):
        face_num = self.face_infos['face_num']
        face_landmarks = self.face_infos['face_landmarks']
        logger.info('face_infos %s', self.face_infos)
        face_complexion_labels: List[str] = self.filter.get_face_complexion(self.image_cv, face_num, face_landmarks)
        print(face_complexion_labels)
        if face_complexion_labels is None or 'black' in face_complexion_labels:  # 人脸肤色识别结果为空或者图像内有黑人
            return True, face_complexion_labels
        logger.info('face_complexion_labels: %s', face_complexion_labels)
        return False, face_complexion_labels

    def test_img_clothes(self):
        """
        服装规则过滤
        """
        # 服装面积过滤
        clothes_fail = True
        clothes_detect_results = self.filter.get_clothes_category_positions(self.image_cv)  # 服装一级标签以及位置信息
        print(clothes_detect_results)
        if len(clothes_detect_results) == 0:  # 无检测结果即无服装
            return False, 'no clothes', ''
        else:
            level_one_labels = []
            for clothes_detect_dict in clothes_detect_results:
                label = clothes_detect_dict['label']
                if label in self.uselessDetectClothesLabels:  # 鞋靴，帽子，包 等标签不进行服装面积阈值设定
                    continue

                level_one_labels.append(label)
                xmin, ymin, xmax, ymax = max(0, float(clothes_detect_dict['xmin'])), max(0, float(
                    clothes_detect_dict['ymin'])), min(1, float(clothes_detect_dict['xmax'])), min(1, float(
                    clothes_detect_dict['ymax']))

                clothes_h, clothes_w = ymax - ymin, xmax - xmin
                # 服装面积过小 或者 过大
                clothes_too_big = clothes_h > 0.9 or clothes_w > 0.9
                clothes_too_tiny = clothes_h < 0.10 or clothes_w < 0.10
                if not (clothes_too_tiny or clothes_too_big):
                    clothes_fail = False
                return False, 'clothes_area unfit', clothes_detect_results
                # if clothes_too_tiny or clothes_too_big:
                #     return False, 'clothes_area unfit', clothes_detect_results
            if clothes_fail:
                return False, 'clothes_area unfit', clothes_detect_results
            # 服装数量过滤
            if len(level_one_labels) == 0:  # 无可用服装
                return False, 'no clothes', []
        return True, 'clothes_filter-ok', clothes_detect_results
