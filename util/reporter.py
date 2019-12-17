import logging
import json
import os
import socket

import pytz
from apscheduler.schedulers.background import BackgroundScheduler

from .common_util import CommonUtil
from .mail import Mail

logger = logging.getLogger(__name__)

hostname = socket.gethostname()
process_id = os.getpid()
max_list_size = os.getenv('MAX_LIST_SIZE', 50)

class Reporter(object):
    def __init__(self):
        self.scheduler = BackgroundScheduler(timezone=pytz.timezone('Asia/Shanghai'))
        self.mail = Mail()
        self.messages = {}
    
    def start_sched(self):
        self.scheduler.start()

    def add_interval_task(self, cb, hours=0, minutes=0, seconds=0):
        self.scheduler.add_job(cb, 'interval', hours=hours, minutes=minutes, seconds=seconds)

    def push_message(self, msg_type, msg):
        if msg_type not in self.messages:
            self.messages[msg_type] = []
        msg_list = self.messages[msg_type]
        msg_list.append(msg)
        if len(msg_list) >= max_list_size:
            logger.info('{} msg list too long [{}], report it ...'.format(msg_type, len(msg_list)))
            self.q_execeed_report(msg_type, msg_list)

    def generate_mail(self, msg_type, msg_list):
        if len(msg_list) > 0:
            subject = '[{}:{}]识别异常通知：{}'.format(hostname, process_id, msg_type)
            miniute_msg_count = {}
            for msg in msg_list:
                minute = CommonUtil.timestamp2str(msg['time'], fmt='%Y%m%d:%H%M')
                count = 0 if minute not in miniute_msg_count else miniute_msg_count[minute]
                miniute_msg_count[minute] = count + 1
            content = miniute_msg_count
            return subject, content
        return None, None

    def q_execeed_report(self, msg_type, msg_list):
        subject, content = self.generate_mail(msg_type, msg_list)
        self.messages[msg_type] = []
        if subject and content:
            subject = '[{}:{}][队列溢出]识别异常通知：{}'.format(hostname, process_id, msg_type)
            logger.info('subject: {}'.format(subject))
            logger.info('content: {}'.format(content))
            self.mail.send_mail(subject=subject, content=json.dumps(content), subtype='plain')

    def interval_report(self):
        logger.info('do reporter interval check ...')
        mail_list = []
        for msg_type in self.messages.keys():
            subject, content = self.generate_mail(msg_type, self.messages[msg_type])
            self.messages[msg_type] = []

            if subject and content:
                mail_list.append([msg_type, content])
        
        if len(mail_list) > 0:
            logger.info('there is something to report ...')
            subject = '[{}:{}][定时上报]识别异常通知：{}'.format(hostname, process_id, 'all type')
            content=json.dumps(mail_list)
            logger.info('subject: {}'.format(subject))
            logger.info('content: {}'.format(content))
            self.mail.send_mail(subject=subject, content=content, subtype='plain')
        else:
            logger.info('there is nothing to report ...')
    