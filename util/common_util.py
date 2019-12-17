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


logger = logging.getLogger(__name__)


class CommonUtil(object):

    @classmethod
    def rand_str(self, len=8):
        salt = ''.join(random.sample(
            string.ascii_letters + string.digits, len))
        return salt

    @classmethod
    def rand_digits_str(self, len=8):
        salt = ''.join(random.sample(''.join(string.digits * 10), len))
        return salt

    @classmethod
    def split_list(self, l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]

    @classmethod
    def md5_url(self, url):
        md5 = hashlib.md5()
        md5.update(url.encode('utf-8'))
        urlmd5 = md5.hexdigest()
        return urlmd5

    @classmethod
    def dump_objects(self, path, *objects):

        logger.info('filename: {}'.format(path))
        try:
            with gzip.open(path, 'wb') as f:
                for obj in objects:
                    cPickle.dump(obj, f, -1)
            return True
        except Exception as e:
            logger.error('dump objects fail: {}'.format(e))
        return False

    @classmethod
    def load_objects(self, path):
        try:
            with gzip.open(path, 'rb') as f:

                return cPickle.load(f)
        except Exception as e:
            logger.error('load object fail: {}'.format(e))
        return None

    @classmethod
    def timestamp2str(self, ts, fmt='%Y-%m-%d %H:%M:%S'):
        return datetime.datetime.fromtimestamp(ts).strftime(fmt)

    @classmethod
    def timestamp2date(self, ts):
        return datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')

    @classmethod
    def executeCmd(self, cmd):
        return subprocess.call(cmd)

    @classmethod
    def executeCmd2(self, cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE):
        return subprocess.Popen(cmd, stdout=stdout, stderr=stderr)

    @classmethod
    def get_exception_info(self):
        return ''.join(traceback.format_exception(*sys.exc_info())[-2:])

