import numpy as np
import cv2
import requests
import logging
import time
import os
import sys
import traceback
import hashlib


logger = logging.getLogger(__name__)

hbase_enable = os.getenv('HBASE_ENABLE', '1')
hbase_rest = 'http://gpu012:8082/image/show-image?rowKey={}'


class DetectBase(object):
    def __init__(self, image_url, image_cv=None, image_pil=None):

        self.image_url = image_url
        self.image_pil = image_pil
        self.image_cv = image_cv
        self.data = None

    @classmethod
    def md5_url(self, url):
        md5 = hashlib.md5()
        md5.update(url.encode('utf-8'))
        urlmd5 = md5.hexdigest()
        return urlmd5

    @classmethod
    def exception_info(cls):
        return ''.join(traceback.format_exception(*sys.exc_info())[-8:])

    def load_image_cv(self):
        """
        先从hbase 下，hbase找不到再从内网
        使用requests
        :param bstype:
        :return:
        """
        if not self.image_cv:
            try:
                time_start = time.time()
                self.data = self.load_image()
                img_array = np.asarray(bytearray(self.data), dtype=np.uint8)
                self.image_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                logger.info('image [{}] cv loaded: {}, cost time: {}'.format(
                    self.image_url, self.image_cv.shape, time.time()-time_start))
            except Exception as e:
                err = 'load image [{}] for cv fail: {}'.format(self.image_url, e)
                logger.exception(err)
                return None, err
        return self.image_cv, None

    def load_image(self):
        """
        我不太习惯这种过程和调用分开，太细了，我都是直接撸
        :return:
        """
        response = self.try_load_hbase_image(self.image_url) if hbase_enable else None
        response = requests.get(self.image_url, timeout=60) if response is None else response

        return response.content

    def try_load_hbase_image(self, url):
        hbase_url = None
        try:
            ori_url = url
            if 'hangzhou-internal.' in url:
                ori_url = url.replace('hangzhou-internal.', 'hangzhou.')
            row_key = self.md5_url(ori_url)
            hbase_url = hbase_rest.format(row_key)
            logger.info('hbase url: {}'.format(hbase_url))
            response = requests.get(hbase_url, timeout=30)
            image_length = int(response.headers['content-length'])
            if image_length > 512:
                logger.info('success to load image from hbase: {}, length: {}'.format(hbase_url, image_length))
                return response
        except Exception as e:
            logger.error('urlopen {} [{}] fail: {}'.format(hbase_url, url, e))
