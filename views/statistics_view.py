import json
import logging

from copy import deepcopy
from datetime import datetime, timedelta

from flask import request
from flask.views import MethodView
from flask import make_response, jsonify, render_template

from typing import Any, Dict, List
from dynaconf import settings
from util.common_service import redis_client


class StatisticsView(MethodView):
    def dispatch_request(self):
        interval = request.args.get('in')
        logging.info('interval=%s', interval)
        print('interval=%s', interval)
        if interval:
            redis_client.hset('Deep_fashion', 'interval', interval)

        return make_response(jsonify({'result': 'change is ok'}), 200)


class GraphView(MethodView):
    def dispatch_request(self, day: int) -> Any:
        now_ts = datetime.strftime(datetime.now()-timedelta(day), '_%Y-%m-%d')
        context: dict = {}
        test = dict(x_data=[], y_data=[])
        result: Dict[str, List] = {}
        today_sum: Dict[str, int] = {}
        interval = int(redis_client.hget('Deep_fashion', 'interval') or 10)
        for i in settings.SERVICE_LIST:
            hash_table = i + now_ts
            service_dic = redis_client.hgetall(hash_table)
            data: List[Dict] = []
            top_data: List[List] = list()
            top_data.append(data)
            logging.info('hash_table: %s,day: %s, redis_client: %s', hash_table, day, redis_client)
            if service_dic:
                print(service_dic)
                tem_ip_sum = {}
                tem_service_sum = {}
                result.update({i: top_data})
                for node, node_dic in service_dic.items():
                    node_dic = json.loads(node_dic)

                    tem_node_sum = {}

                    for ip, ip_dic in node_dic.items():
                        lst = []
                        data.append({'name': node+'_'+ip[-5:], 'data': lst})

                        for period, call_times in ip_dic.items():
                            print(period,'-'*10)
                            ts = now_ts + ' ' + period.split('-')[0]
                            print(ts, 'ts-' * 10)
                            # tm = datetime.strftime(datetime.strptime(ts, '_%Y-%m-%d %H:%M'), "%m-%d %H:%M")
                            tm = datetime.strptime(ts, '_%Y-%m-%d %H:%M').timestamp()*1000
                            tem_node_sum[tm] = tem_node_sum[tm] + call_times if tem_node_sum.get(tm) else call_times

                            tem_ip_dic = tem_ip_sum.get(ip) if tem_ip_sum.get(ip) else {}
                            tem_ip_sum.update({ip: tem_ip_dic})

                            tem_ip_dic[tm] = tem_ip_dic[tm] + call_times if tem_ip_dic.get(tm) else call_times
                            tem_service_sum[tm] = tem_service_sum[tm] + call_times if tem_service_sum.get(tm) else call_times
                            lst.append([tm, call_times])


                            # test['x_data'].append(tm)
                            # test['y_data'].append(call_times)
                    # 对一个服务的各个节点,及各个IP进行汇总
                    # for k, v in tem_node_sum.items():
                    #     node_sum.append([k, v])
                    node_sum = [[k, v] for k, v in tem_node_sum.items()]

                    data.append({'name': node, 'data': node_sum})
                    print('tem_node_sum', tem_node_sum)
                    print('node_sum', node_sum)
                for ip, ip_sum in tem_ip_sum.items():
                    ip_sum = [[k, v] for k, v in ip_sum.items()]
                    data.append({'name': ip, 'data': ip_sum})

                service_sum = [[k, v] for k, v in tem_service_sum.items()]
                data.append({'name': i, 'data': service_sum})
                new = deepcopy(data)
                # [i.update(type='column') for i in new]
                [j.update(type='column') for j in new if j['name'] == i]
                column: List[Dict] = [j for j in new if j['name'] == i]
                # [j.update(type='column') for j in data if j['name'] == i]
                # data.extend(new)
                print('column', column)
                top_data.append(column)
                print('today_sum', today_sum)
                today_sum.update({i: sum(i for i in tem_service_sum.values())})

                print('tem_ip_sum', tem_ip_sum)
        print('today_sum', today_sum)
        logging.info('result: %s', result)
        context.update(test=test)
        context.update(interval=interval)
        context.update(today_sum=today_sum)
        context.update(result=result)
        return render_template('test.html', **context)