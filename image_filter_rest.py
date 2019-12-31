from flask import Flask

import logging

from src.filter_lib.image_filter import ImageFilterConstructor
from views.image_filter_view import ImageFilterView
from util.statistics import start_daemon_thread

from service.logging_service import logger

app = Flask(__name__)
# 支持返回结果显示中文
app.config['JSON_AS_ASCII'] = False

image_filter = ImageFilterConstructor.instance()

ImageFilterView.init_version()
ImageFilterView.init_filter(image_filter)

app.add_url_rule('/image/filter', view_func=ImageFilterView.as_view('image_filter'))


@app.route('/', endpoint='index')
def hello_world():
    return 'Hello World!'


@app.before_first_request
def daemon_thread():
    start_daemon_thread()


if __name__ == "__main__":
    logger.info('image filter detect service start ...')

    # start_daemon_thread()
    app.run(host="0.0.0.0", port=9003, threaded=True)
