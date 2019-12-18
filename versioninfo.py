import os
import json
import logging

logger = logging.getLogger(__name__)

class VersionInfo(object):

    def __init__(self):
        self.version_file = "./version.info"
        self.model_version = {
            "filter_model_version": "20180607",
            "category_model_version": "old",
            "clothes_model_version": "20180702"
        }

    def get_model_version(self):
        
        if os.path.exists(self.version_file):
            try:
                with open(self.version_file) as f:
                    lines = f.readlines()
                    info = ''.join([l.strip() for l in lines])
                    self.model_version = json.loads(info)
                    logger.info('version info: {}'.format(self.model_version))
            except Exception as e:
                logger.error('load version info fail: {}'.format(e))
        return self.model_version

    def get_filter_model_version(self):
        return self.model_version['filter_model_version']

    def get_category_model_version(self):
        return self.model_version['category_model_version']