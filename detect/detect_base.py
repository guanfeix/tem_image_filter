import numpy as np
import cv2
import requests
import logging
import time
import os
import sys
import traceback
import hashlib

from urllib.request import urlopen
from io import BytesIO
from PIL import Image


logger = logging.getLogger(__name__)

hbase_hosts = os.getenv('HBASE_HOSTS', 'cpu005,cpu006,cpu007').split(',')
hbase_hosts_size = len(hbase_hosts)
hbase_rest = 'http://{}:8806/image/{}/{}'
hbase_enable = os.getenv('HBASE_ENABLE', '1')

bstype_dict = {
    'ins': 'ins',
    'instagram': 'ins',
    'pop': 'pop',
    'show': 'show',
    'pop_show': 'show',
    'vogue_runway': 'show',
    'runway': 'show',
    'market': 'market',
    'diotion': 'diotion',
}

bstype_list = ['ins', 'show', 'market', 'diotion', 'pop']

class DetectBase(object):
    def __init__(self, image_url, image_cv=None, image_pil=None):

        self.image_url = image_url
        self.image_pil = image_pil
        self.image_cv = image_cv

    @classmethod
    def md5_url(self, url):
        md5 = hashlib.md5()
        md5.update(url.encode('utf-8'))
        urlmd5 = md5.hexdigest()
        return urlmd5

    @classmethod
    def exception_info(self):
        return ''.join(traceback.format_exception(*sys.exc_info())[-8:])

    def load_image_pil(self):

        if not self.image_pil:
            try:
                r = requests.get(self.image_url, timeout=60)
                self.image_pil = Image.open(BytesIO(r.content))
                logger.info('image pil loaded: {}'.format(self.image_pil))
            except Exception as e:
                logger.error('load image for pil fail: {}'.format(e))
        return self.image_pil

    def load_image_cv(self, bstype=None):

        if not self.image_cv:
            try:
                time_start = time.time()
                request = self.try_load_idc_image_cv(self.image_url, bstype) if hbase_enable else None
                resize_url = '{}?{}'.format(self.image_url, 'x-oss-process=image/resize,l_1000')
                request = urlopen(resize_url, timeout=60) if request == None else request
                img_array = np.asarray(
                    bytearray(request.read()), dtype=np.uint8)
                self.image_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                logger.info('image [{}] cv loaded: {}, cost time: {}'.format(
                    self.image_url, self.image_cv.shape, time.time()-time_start))
            except Exception as e:
                err = 'load image [{}] for cv fail: {}'.format(self.image_url, e)
                logger.error(err)
                return None, err
        return self.image_cv, None

    @classmethod
    def get_bstype(self, url):
        
        for k in bstype_dict.keys():
            if k in url:
               return bstype_dict[k]
        return 'ins'

    def try_load_idc_image_cv(self, url, bstype=None):

        hbase_url = None
        try:
            bstype = bstype if bstype else self.get_bstype(url)
            ori_url = url
            if 'hangzhou-internal' in url:
                ori_url = url.replace('hangzhou-internal', 'hangzhou')
            time_start = time.time()
            hbase_host = hbase_hosts[int(time_start) % hbase_hosts_size]
            rowkey = self.md5_url(ori_url)
            request_tables = [bstype]
            [request_tables.append(x) for x in bstype_list if x != bstype]
            for table in request_tables:
                hbase_url = hbase_rest.format(hbase_host, table, rowkey)
                logger.info("urlopen url: {}".format(hbase_url))
                request = urlopen(hbase_url, timeout=30)
                image_length = int(request.headers['content-length'])
                logger.info('load image from idc, image length: {}, cost time: {}'.format(
                    image_length, time.time() - time_start))
                if image_length > 512:
                    logger.info('download image from idc: {}'.format(hbase_url))
                    return request
        except Exception as e:
            logger.error('urlopen {} [{}] fail: {}'.format(hbase_url, url, e))
        
        return None

    def cut_images_simple(self, image_cv, positions):
        img_h, img_w, _ = image_cv.shape
        imgs = []
        for position in positions:
            xmin, ymin, xmax, ymax = float(position['xmin']), float(
                position['ymin']), float(position['xmax']), float(position['ymax'])
            xmin, ymin, xmax, ymax = int(
                xmin*img_w), int(ymin*img_h), int(xmax*img_w), int(ymax*img_h)
            crop_img = image_cv[ymin:ymax, xmin:xmax]
            imgs.append(crop_img)
        return imgs

    def cut_images_complex(self, image_cv, positions):
        img_h, img_w, _ = image_cv.shape
        imgs = []
        for position in positions:
            xmin, ymin, xmax, ymax = float(position['xmin']), float(position['ymin']), float(position['xmax']), float(position['ymax'])
            crop_img = self.cut_image(image_cv, [xmin, ymin, xmax, ymax])
            imgs.append(crop_img)
        return imgs

    def judge_w_h(self, center, box_position, box_h_w, img_h_w, amplify=0.05):
        """判断高大于宽的情况"""
        img_h, img_w = box_h_w[0], box_h_w[1]
        x_min, y_min, x_max, y_max = box_position[0], box_position[1], box_position[2], box_position[3]
        image_h, image_w = img_h_w[0], img_h_w[1]

        img_h = (1 + amplify) * img_h
        y_min, y_max = max(int(y_min - amplify * img_h), 0), min(int(y_max + amplify * img_h), image_h)

        if int(img_h / img_w) > 2:  # 长宽比大于3
            w = int((img_h / 3 * 2) / 2)
            x_min, y_min, x_max, y_max = max((center[0] - w), 0), y_min, min((center[0] + w), image_w), y_max
        else:  # 长宽比小于3
            w = int(img_h / 2)
            x_min, y_min, x_max, y_max = max((center[0] - w), 0), y_min, min((center[0] + w), image_w), y_max
        return x_min, y_min, x_max, y_max

    def judge_h_w(self, center, box_position, box_h_w, img_h_w):
        """判断宽大于高的情况"""
        img_h, img_w = box_h_w[0], box_h_w[1]
        x_min, y_min, x_max, y_max = box_position[0], box_position[1], box_position[2], box_position[3]
        image_h, image_w = img_h_w[0], img_h_w[1]

        if int(img_w / img_h) > 2:  # 长宽比大于3
            h = int((img_w / 3 * 2) / 2)
            x_min, y_min, x_max, y_max = x_min, max((center[1] - h), 0), x_max, min((center[1] + h), image_h)
        else:
            h = int(img_w / 2)
            x_min, y_min, x_max, y_max = x_min, max((center[1] - h), 0), x_max, min((center[1] + h), image_h)
        return x_min, y_min, x_max, y_max

    def cut_image(self, image, position):
        """数据裁剪"""
        xmin, ymin, xmax, ymax = position
        image_h, image_w, _ = image.shape  # image_h, image_w 分别表示整图的大小
        x_min, y_min, x_max, y_max = max(int(xmin * image_w), 0), \
                                    max(int(ymin * image_h), 0), \
                                    min(int(xmax * image_w), image_w), \
                                    min(int(ymax * image_h), image_h)
        img_h, img_w = (y_max - y_min), (x_max - x_min)  # img_h, img_w 分别表示box的长和宽
        box_position = [x_min, y_min, x_max, y_max]
        box_h_w = [img_h, img_w]
        img_h_w = [image_h, image_w]
        if img_h < 0 or img_w < 0:
            print('the error box:', position)
        center = [x_min + int(img_w / 2), y_min + int(img_h / 2)]
        if img_h > img_w:
            x_min, y_min, x_max, y_max = self.judge_w_h(center, box_position, box_h_w, img_h_w, amplify=0.05)
        else:
            x_min, y_min, x_max, y_max = self.judge_h_w(center, box_position, box_h_w, img_h_w)

        cut_img = image[y_min:y_max, x_min:x_max]
        return cut_img