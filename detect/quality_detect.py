import numpy as np
import cv2
from urllib.request import urlopen

import sys
import traceback
import logging

from .detect_base import DetectBase

logger = logging.getLogger(__name__)


class QualityDetect(DetectBase):

    def __init__(self, filter, image_url, image_source='weibo',image_cv=None):

        super(QualityDetect, self).__init__(image_url, image_cv)
        self.filter = filter
        self.face_infos = None
        self.image_source = image_source

    def get_image_shape(self):
        if self.image_cv:
            return self.image_cv.shape
        
        return None, None, None

    def is_gray_image(self):
        gray = self.filter.is_img_gray(self.image_cv)
        return gray

    def get_image_resolution(self):
        clear, resolution = self.filter.cal_img_resolution(self.image_cv)
        return not clear, resolution

    def get_image_brightness(self):
        brightness = self.filter.cal_img_brightness(self.image_cv)
        return brightness

    def get_image_faceinfo(self):
        # num, positions, landmarks = self.filter.get_img_face_info(self.image_cv)
        num, positions, landmarks = self.filter.get_face_positions(self.image_cv)
        if positions:
            for pos in positions:
                pos[0], pos[1], pos[2], pos[3] = (int(pos[0]), int(pos[1]),
                                                  int(pos[2]), int(pos[3]))
        logger.info('face number: {}, positions: {}'.format(num, positions))
        return num, positions, landmarks

    def get_face_blur(self, pos):
        xmin, ymin, xmax, ymax = pos[0], pos[1], pos[2], pos[3]
        blur, resolution = self.filter.cal_img_resolution(
            self.image_cv[ymin:ymax, xmin:xmax])
        return not blur, resolution

    def get_face_attributes(self):

        self.prepare_faceinfos()

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

    def prepare_faceinfos(self):

        try:
            if not self.face_infos:
                img_h, img_w, img_c = self.image_cv.shape
                img_w_f, img_h_f = float(img_w), float(img_h)
                face_num, face_positions, face_landmarks = self.get_image_faceinfo()
                face_normal_positions = []
                face_positions = face_positions if face_num > 0 else []
                for index, position in enumerate(face_positions):
                    xmin, ymin, xmax, ymax = position[0], position[1], position[2], position[3]
                    face_pos = {
                        "xmin": round(xmin / img_w_f, 2),
                        "ymin": round(ymin / img_h_f, 2),
                        "xmax": round(xmax / img_w_f, 2),
                        "ymax": round(ymax / img_h_f, 2)
                    }
                    face_normal_positions.append(face_pos)

                self.face_infos = {
                    'face_num': face_num,
                    'face_positions': face_positions,
                    'face_landmarks': face_landmarks,
                    'face_normal_positions': face_normal_positions
                }
        except Exception as e:
            logger.error('prepare face info fail: {}'.format(e))
            logger.error(''.join(traceback.format_exception(*sys.exc_info())[-2:]))

    def check_face_position(self, pos):
        img_h, img_w, img_c = self.image_cv.shape
        xmin, ymin, xmax, ymax = pos[0], pos[1], pos[2], pos[3]
        face_w, face_h = xmax - xmin, ymax - ymin

        is_big = face_h > 0.25 * img_h
        is_small = face_h < 0.05 * img_h or face_w < 40 or face_h < 40
        is_center = (xmin > 0.15 * img_w and xmin < 0.85 *
                     img_w) and (ymin > 0 and ymax < 0.85 * img_h)
        return is_big, is_small, is_center

    def judge_image_shape(self):
        img_h, img_w, _ = self.image_cv.shape
        if self.image_source == 'ins':
            return min(img_h, img_w) > 720

        return min(img_h, img_w) > 640 and max(img_h, img_w) < 1800

    def tag_base_info(self):

        base_tags = []

        is_gray = self.is_gray_image()
        if is_gray:
            base_tags.append('filter-gray')

        image_shape, flag = self.judge_image_shape()

        blur, resolution = self.get_image_resolution()
        if blur:
            base_tags.append('filter-blur')

        brightness = self.get_image_brightness()
        if brightness == 'dark':
            base_tags.append('filter-brightness-dark')

        return base_tags, {'image-blur': blur, 'image-resolution': resolution, 'image-shape': self.image_cv.shape}

    def tag_faces_info(self, max_face_num=2):

        self.prepare_faceinfos()
        face_num = self.face_infos['face_num']
        if face_num == 0 or face_num > max_face_num:
            return False, ['filter-face-num-manyornone'], None

        faceinfo_tags = []
        face_positions = self.face_infos['face_positions']
        face_normal_positions = self.face_infos['face_normal_positions']
        face_landmarks = self.face_infos['face_landmarks']
        for index, position in enumerate(face_positions):
            status, tag = self.tag_face_info(
                index, position, face_normal_positions[index], face_landmarks[:, index])
            if status:
                faceinfo_tags.append(tag)
            else:
                return False, [tag], None

        return True, None, faceinfo_tags

    def tag_face_info(self, index, position, normal_pos, landmarks):

        fail_face_tag = {'face_index': index,
                    'face_tags': []}

        is_big, is_small, is_center = self.check_face_position(position)
        if is_big or is_small:
            fail_face_tag['face_tags'].append('filter-face-bigorsmall')
        if not is_center:
            fail_face_tag['face_tags'].append('filter-face-nocenter')

        if is_big or is_small or not is_center:
            return False, fail_face_tag

        blur, resolution = self.get_face_blur(position)
        if blur:
            fail_face_tag['face_tags'].append('filter-face-blur')
            return False, fail_face_tag

        faceinfo = {"face_index": index,
                    "face_resolution": resolution,
                    "face_blur": blur}
        landmark_list = landmarks.tolist()
        face_lms_len = len(landmark_list)
        landmark_info = {}
        img_h, img_w, img_c = self.image_cv.shape
        img_w_f, img_h_f = float(img_w), float(img_h)
        for i in range(0, 5):
            landmark_info["xmin%d" % i] = round(landmark_list[i] / img_w_f, 2)
            landmark_info["ymin%d" % i] = round(
                landmark_list[i+5] / img_h_f, 2)
        faceinfo['landmark'] = landmark_info
        faceinfo['postion'] = normal_pos

        return True, faceinfo

    def tag_face_attributes(self):

        complexion_list, sex_list, race_list = self.get_face_attributes()
        landmarks = self.face_infos['face_landmarks']
        positions = self.face_infos['face_normal_positions']

        faceattr_info_list = []
        black_face = False
        for position, complexion, sex, race in zip(positions, complexion_list, sex_list, race_list):

            face_tags = []
            if complexion == "black":
                face_tags.append("filter-face-black")
                black_face = True
            elif complexion == "not_black":
                face_tags.append("filter-face-notblack")

            if sex == "woman":
                face_tags.append("filter-face-woman")
            elif sex == "man":
                face_tags.append("filter-face-man")

            if race == "asia":
                face_tags.append("filter-face-asia")
            elif race == "europe":
                face_tags.append("filter-face-europe")

            if len(face_tags) > 0:

                faceattr_info = {
                    'position': position,
                    'tags': face_tags,
                    "precision": 1
                }
                faceattr_info_list.append(faceattr_info)
        return black_face, faceattr_info_list