from flask import Flask
from flask import request
from flask import make_response, jsonify
from flask.views import View

import traceback
import sys
import time
import json
import logging

logger = logging.getLogger(__name__)

from .detect_view import DetectView
from versioninfo import VersionInfo
from detect.quality_detect import QualityDetect
from util.statistics import statistics_recognize

import socket
hostname = socket.gethostname()


class ImageFilterView(DetectView):

    methods = ['POST']
    decorators = [statistics_recognize('ImageFilter')]
    # @statistics_recognize('ImageFilter')

    def dispatch_request(self):
        result = {
            'server_name': hostname,
            'filter_result': False,
            'filter_model_version': self.version.get_filter_model_version(),
        }

        filter_tag_list = dict()
        result['filter_tags'] = filter_tag_list
        base_tag_list = []
        filter_tag_list['base_tags'] = base_tag_list

        try:
            data = request.get_data()
            json_data = json.loads(data)
            logger.info('request params: {}'.format(json_data))

            image_url = json_data['imageUrl']
            image_source = json_data['image_source']
            result['url'] = image_url

            detect = QualityDetect(self.image_filter, image_url, image_source)
            detect.load_image_cv()

            base_tags, resolution_tag = detect.tag_base_info()
            logger.info('base info tags:{}, resolution: {}'.format(base_tags, resolution_tag))
            [base_tag_list.append(tag) for tag in base_tags]
            filter_tag_list['blur-resolution'] = resolution_tag

            is_ok, fail_tags, face_tags = detect.tag_faces_info()
            logger.info('face info is_ok: {}, fail_tags: {}, face_tags: {}'.format(is_ok, fail_tags, face_tags))
            if not is_ok:
                [base_tag_list.append(tag) for tag in fail_tags]

                logger.info('result: {}'.format(result))
                return make_response(jsonify(result), 200)

            face_info_list = []
            filter_tag_list['face_infos'] = face_info_list
            for tags in face_tags:
                face_info_list.append(tags)
            
            black_face, faceattr_tags = detect.tag_face_attributes()
            for i, tags in enumerate(faceattr_tags):
                face_tags[i]['tags'] = tags['tags']
            
            if black_face:
                base_tag_list.append('filter-face-black')

            quality_ok = len(base_tag_list) == 0 and not black_face
            if quality_ok:
                base_tag_list.append('filter-quality-ok')
                result['filter_result'] = True 

            logger.info('result: {}'.format(result))
            return make_response(jsonify(result), 200)

        except Exception as e:
            logger.error('image filter fail: {}'.format(e))
            logger.error(''.join(traceback.format_exception(*sys.exc_info())[-2:]))
            result['error'] = '{}'.format(e)
            return make_response(jsonify(result), 500)