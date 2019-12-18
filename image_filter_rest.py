import os
import sys
sys.path.append(os.path.abspath("../src"))
from flask import Flask

import logging

from src.filter_lib.image_filter_lib import ImageFilter
# from color_lib.imgColor import genImgColor

from views.image_filter_view import ImageFilterView
# from views.detect_gray_view import DetectGrayView
# from views.detect_brightness_view import DetectBrightnessView
# from views.detect_resolution_view import DetectResolutionView
# from views.detect_faceinfo_view import DetectFaceinfoView
# from views.detect_faceattr_view import DetectFaceattrView
# from views.detect_color_view import DetectColorView
from util.statistics import start_daemon_thread


logging.basicConfig(
    level=logging.INFO,
    format="[%(thread)d] [%(asctime)s] %(name)s:%(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)

app = Flask(__name__)
# 支持返回结果显示中文
app.config['JSON_AS_ASCII'] = False

image_filter = ImageFilter.instance()
# color_detect = genImgColor()

ImageFilterView.init_version()
ImageFilterView.init_filter(image_filter)
# ImageFilterView.init_color_detect(color_detect)

app.add_url_rule('/image/filter', view_func=ImageFilterView.as_view('image_filter'))
# app.add_url_rule('/image/detect/gray', view_func = DetectGrayView.as_view('detect_gray'))
# app.add_url_rule('/image/detect/brightness', view_func = DetectBrightnessView.as_view('detect_brightness'))
# app.add_url_rule('/image/detect/resolution', view_func = DetectResolutionView.as_view('detect_resolution'))
# app.add_url_rule('/image/detect/faceinfo', view_func = DetectFaceinfoView.as_view('detect_faceinfo'))
# app.add_url_rule('/image/detect/faceattr', view_func = DetectFaceattrView.as_view('detect_faceattr'))
# app.add_url_rule('/image/detect/color', view_func = DetectColorView.as_view('detect_color'))


@app.before_first_request
def daemon_thread():
    start_daemon_thread()


if __name__ == "__main__":
    logger.info('image filter detect service start ...')

    # start_daemon_thread()
    app.run(host="0.0.0.0", port=9002, threaded=True)
