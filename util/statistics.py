import time
import ssl
import json
import functools
import threading

import logging as logger
from queue import Queue, Empty
from datetime import datetime, timedelta

from flask import request

from util.common_util import CommonUtil
from util.common_service import redis_client, host_name, get_running_port

logger = logger.getLogger(__name__)


def statistics_recognize(service_name: str):
    def decorate(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            task = dict(service_name=service_name, now=datetime.now(), ip=request.remote_addr)
            logger.info('receive statistics task: {} service was called'.format(service_name))
            logger.info('request_host:%s,request.environ: %s ,request.remote_addr= %s, user_agent=%s',
                        request.host, request.remote_user, request.remote_addr, request.user_agent)
            if not StatisticsThread.is_full():
                StatisticsThread.put_task(task)
            return func(*args, **kwargs)
        return wrapper
    return decorate


def start_daemon_thread():
    ssl._create_default_https_context = ssl._create_unverified_context
    worker = StatisticsThread()
    worker.name = 'daemon-statistics-loop'
    worker.daemon = True
    worker.start()


class StatisticsThread(threading.Thread):
    """

    """
    # 放外面不生效，更改
    interval = int(redis_client.hget('Deep_fashion', 'interval') or 10)
    statis_tasks_q = Queue(maxsize=1000)

    @classmethod
    def put_task(cls, task):
        try:
            cls.statis_tasks_q.put_nowait(task)
        except Exception as e:
            logger.error('put task to queue fail, full: {}'.format(cls.statis_tasks_q.full()))

    @classmethod
    def get_task(cls, timeout=None):
        return cls.statis_tasks_q.get(block=True, timeout=timeout)

    @classmethod
    def is_full(cls):
        return cls.statis_tasks_q.full()

    @property
    def qsize(self):
        return self.statis_tasks_q.qsize()

    # def statistics_compute(self, task_json):
    #     """
    #     延时写入，一个时间内总次数，减少Redis的访问次数
    #     :param service_name:
    #     :return:
    #     """
    #     now = task_json.get('now', datetime.now())
    #     for s, t in task_json.items():
    #         self._count(s, now, t)

    def run(self):
        lst = []
        timer = retry = 0
        logger.info('models statistics daemon running ...')
        while True:
            try:
                try:
                    task = self.get_task(timeout=0.01)
                except Empty as e:
                    logger.error('Error in get_task from Queue: %s', e)
                    timer += 2
                    time.sleep(timer if timer < 50 else 50)
                else:
                    lst.append(task)
                    timer = 0

                now = datetime.now()
                logger.debug('lst=%s,now.minute %sinterval=%s' % (len(lst), now.minute, now.minute % self.interval == 0))
                # if now.minute % self.interval == 0 and lst or len(lst) > 1000 or timer > 600: # timer没有意义
                if now.minute % self.interval == 0 and lst or len(lst) > 1000:
                    # 三种情况：正常-时间间隔和快-累积量,慢的话就阻塞时间
                    if self.qsize > 0 and timer == 0 and retry < 900:
                        # 确保出发处理略过的是累计的任务，
                        retry += 1
                        continue
                    logger.info('handle task: {}'.format(len(lst)))
                    self.test_count(lst)
                    self.refresh_interval()
                    lst.clear()
                    timer = retry = 0
                    time.sleep(60)
            except Exception as e:
                logger.exception('exception: {}'.format(CommonUtil.get_exception_info()))
                logger.info('Error,handwriting=%s', e)

    @classmethod
    def test_count(cls, lst):
        """
        感觉增加了一个缓存写入，复杂度上升了一个级别
        :param lst:
        :return:
        """
        result = {}
        now = lst[0].get('now')
        end_now = lst[-1].get('now')
        # 临界的时间，没有处理,第二天开始的统计，会被加到第一天
        if end_now.day != now.day:
            result.update({datetime.strftime(end_now, '_%Y-%m-%d'): {}})

        result.update({datetime.strftime(now, '_%Y-%m-%d'): {}})
        logger.info('lst: %s', len(lst))

        for task in lst:
            # 统计各时间间隔的频次
            service_name = task.get('service_name')
            ip = task.get('ip')
            now = task.get('now')
            now_while = str((now.minute // cls.interval) * cls.interval).zfill(2)
            next_while = (now.minute // cls.interval + 1) * cls.interval
            # todo 对应的统计处该为[1]，换成时间间隔后一节点会出现跨区问题，也就是12：60应该写成1：00，
            #  也就是会有个进位时间间隔换成时间点本身就不太合适
            next_while = str(next_while if next_while <= 60 else 60).zfill(2)
            now_tm = datetime.strftime(now, '_%Y-%m-%d')
            period = '%d:%s-%d:%s' % (now.hour, now_while, now.hour, next_while)
            tem_result = result.get(now_tm)

            if service_name not in tem_result.keys():
                tem_result.update({service_name: {}})
            service_dic = tem_result.get(service_name)
            if ip not in service_dic.keys():
                service_dic.update({ip: {}})
            ip_statistic = service_dic.get(ip)
            ip_statistic[period] = ip_statistic[period]+1 if ip_statistic.get(period) else 1

        logger.info('service count：%s', result)
        port = get_running_port()
        node = host_name + ':' + str(port)

        for now_ts, dic in result.items():
            for service_name, service_dic in dic.items():
                hash_table = service_name + now_ts
                node_statis = redis_client.hget(hash_table, node, ) or str({})
                logger.debug('node_statis %s', node_statis)
                redis_preiod_dic = json.loads(node_statis)

                for ip_name, ip_dic in service_dic.items():
                    redis_ip_dic = redis_preiod_dic.get(ip_name)
                    if redis_ip_dic:
                        for k, v in ip_dic.items():
                            redis_ip_dic[k] = (redis_ip_dic.get(k) or 0) + v
                    else:
                        redis_preiod_dic.update({ip_name: ip_dic})
                print('redis_preiod_dic', redis_preiod_dic)
                redis_client.hset(hash_table, node, json.dumps(redis_preiod_dic))
                redis_client.expire(hash_table,timedelta(days=30))
                logger.info('hash_table: {}, field: {}, new_dic: {}'.format(hash_table, node, service_dic))


    def refresh_interval(self):
        self.interval = int(redis_client.hget('Deep_fashion', 'interval') or 10)
        logger.info('Interval was refresh,interval={}'.format(self.interval))
