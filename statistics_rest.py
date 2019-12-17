import sys
import logging
from flask import Flask, request
from flask import make_response, jsonify

# sys.path.append('..')
print(sys.path)

from util.common_service import redis_client
from views.statistics_view import StatisticsView, GraphView


app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

logging.basicConfig(
    level=logging.INFO,
    format="[%(thread)d] [%(asctime)s] %(name)s:%(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


@app.route('/statistics/interval', methods=['POST', 'GET'])
def change_interval():
    """
    这里统计间隔最好是60以内的分钟数,公约数
    :return:
    """
    if request.method == "GET":
        interval = request.args.get('in')
        logger.info('interval=%s', interval)
        print('interval=%s', interval)
        if interval:
            redis_client.hset('Deep_fashion', 'interval', interval)
            return make_response(jsonify({'result': 'change is ok'}), 200)

        return '<h1>Bad Request：No Change</h1>', 400


app.add_url_rule('/statistics/int', view_func=StatisticsView.as_view('statistics_interval'))
app.add_url_rule('/graph/<int:day>', view_func=GraphView.as_view('graphic_show'))
app.add_url_rule('/graph/', view_func=GraphView.as_view('graphic_show1'), defaults={'day': 0})


@app.route('/test', endpoint='Test')
def test():
    pass


@app.route('/', endpoint='index')
def hello_world():
    return 'Hello World!'

if __name__ == '__main__':
    # print(app.url_map)
    app.run(host="0.0.0.0", port=9011, threaded=True, debug=True)