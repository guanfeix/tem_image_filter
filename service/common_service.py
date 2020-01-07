import os
import string
import time
import datetime
import requests
import random
import _pickle as cPickle
import gzip
import logging
import hashlib
import traceback
import sys
import subprocess

from dynaconf import settings
from datetime import datetime, timedelta
from typing import List, Generator, Iterator

from service.logging_service import logger


class CommonUtil(object):

    @staticmethod
    def rand_str(len=8):
        salt = ''.join(random.sample(
            string.ascii_letters + string.digits, len))
        return salt

    @staticmethod
    def rand_digits_str(length=8):
        salt = ''.join(random.sample(''.join(string.digits * 10), len))
        return salt

    @staticmethod
    def split_list(l: List, n: int) -> Iterator[List]:
        for i in range(0, len(l), n):
            yield l[i:i + n]

    @staticmethod
    def md5_url(url):
        md5 = hashlib.md5()
        md5.update(url.encode('utf-8'))
        urlmd5 = md5.hexdigest()
        return urlmd5

    @staticmethod
    def md5_bytes(data):
        md5 = hashlib.md5()
        md5.update(data)
        md5str = md5.hexdigest()
        return md5str

    @staticmethod
    def dump_objects(path, *objects):

        logger.info('filename: {}'.format(path))
        try:
            with gzip.open(path, 'wb') as f:
                for obj in objects:
                    cPickle.dump(obj, f, -1)
            return True
        except Exception as e:
            logger.error('dump objects fail: {}'.format(e))
        return False

    @staticmethod
    def load_objects(path):
        try:
            with gzip.open(path, 'rb') as f:

                return cPickle.load(f)
        except Exception as e:
            logger.error('load object fail: {}'.format(e))
        return None

    @staticmethod
    def deltaDays2date(delta, fmt='%Y%m%d'):
        days_before = datetime.now() - timedelta(days=delta)
        return days_before.strftime(fmt)

    @staticmethod
    def deltaMinutes2date(delta, fmt='%Y%m%d'):
        days_before = datetime.now() - timedelta(min=delta)
        return days_before.strftime(fmt)

    @staticmethod
    def timestamp2str(ts, fmt='%Y-%m-%d %H:%M:%S'):
        return datetime.fromtimestamp(ts).strftime(fmt)

    @staticmethod
    def timestamp2date(ts):
        return datetime.fromtimestamp(ts).strftime('%Y-%m-%d')

    @staticmethod
    def executeCmd(cmd):
        return subprocess.call(cmd)

    @staticmethod
    def executeCmd2(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE):
        return subprocess.Popen(cmd, stdout=stdout, stderr=stderr)

    @staticmethod
    def get_exception_info(lines=-2):
        return ''.join(traceback.format_exception(*sys.exc_info())[lines:])


def pick_request_url(url):
    if settings.OSSURL_INTERNAL and 'oss-cn-hangzhou-internal' not in url:
        url = url.replace('oss-cn-hangzhou', 'oss-cn-hangzhou-internal')
        logger.info('url: %s', url)
    return url
