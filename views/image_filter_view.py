import sys
import socket
import traceback

from typing import Dict, List, Any

from flask import request
from flask import make_response, jsonify

from dynaconf import settings

from views.detect_view import DetectView
from detect.quality_detect import QualityDetect
from util.statistics import statistics_recognize
from service.common_service import pick_request_url

from service.logging_service import logger
hostname = socket.gethostname()
duplicate = settings.DEDUPLICATE


class ImageFilterView(DetectView):

    methods = ['POST']
    decorators = [statistics_recognize('TemImageFilter')]

    def dispatch_request(self):
        result = {
            'server_name': hostname,
            'filter_result': False,
            'filter_model_version': self.version.get_filter_model_version(),
        }
        base_tag_list: List[str] = []
        face_info_list: List[Dict] = []
        filter_tag_list: Dict[str, Any] = dict()
        result['filter_tags'] = filter_tag_list
        filter_tag_list['base_tags'] = base_tag_list
        filter_tag_list['face_infos'] = face_info_list
        try:
            json_data = request.get_json()
            logger.info('request params: {}'.format(json_data))
            url = json_data['imageUrl']
            image_source = json_data['image_source']
            result['url'] = url
            # 图片载入
            detect = QualityDetect(self.image_filter, pick_request_url(url), image_source)
            detect.load_image_cv()
            # 基本信息
            base_tags, resolution_tag = detect.tag_base_info()

            logger.info('base info tags:{}, resolution: {}'.format(base_tags, resolution_tag))
            [base_tag_list.append(tag) for tag in base_tags]
            filter_tag_list['blur-resolution'] = resolution_tag
            # 人脸过滤
            is_face_ok, fail_tag, face_normal_positions, face_num = detect.tag_faces_info()

            logger.info('face info is_ok: {}, fail_tags: {}, face_normal_positions: {}'.
                        format(is_face_ok, fail_tag, face_normal_positions))
            # 记录面部信息
            for index, pos in enumerate(face_normal_positions):
                face_info_list.append({"face_index": index, "position": pos})

            # 黑人图片去除
            if face_num > 0:
                black_face, face_complexion_list = detect.get_face_complexion()
                # 记录每张脸的肤色信息
                for i, face in enumerate(face_info_list):
                    face['complexion'] = face_complexion_list[i]
                if black_face:
                    base_tag_list.append('filter-face-black')
            # 衣物过滤
            is_ok, msg, clothes_detect_results = detect.test_img_clothes()
            filter_tag_list['clothes_infos'] = clothes_detect_results
            # 文本过滤
            text_exist = detect.tag_text_info()

            # 去重检测
            is_dup = False
            if duplicate:
                hash_result, dups = detect.filter_dedup_image(url)
                is_dup = True if dups else False
                result['duplicated'] = is_dup
                result['duplicated_features'] = dups
                result['feature_hash']: int = hash_result[1]

            clothes_only_ok = is_ok and not text_exist and not is_dup
            if not is_face_ok:
                base_tag_list.append(fail_tag)
            if not is_ok:
                base_tag_list.append(msg)
            if text_exist:
                base_tag_list.append('text_exist')

            quality_fail = base_tag_list or is_dup

            if not quality_fail:
                base_tag_list.append('filter-quality-ok')
                result['filter_result'] = True 
            result['filter_result_V2'] = clothes_only_ok
            logger.info('result: {}'.format(result))
            return make_response(jsonify(result), 200)

        except Exception as e:
            logger.error('image filter fail: {}'.format(e))
            logger.error(''.join(traceback.format_exception(*sys.exc_info())[-2:]))
            result['error'] = '{}'.format(e)
            return make_response(jsonify(result), 500)