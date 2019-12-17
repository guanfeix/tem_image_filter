# -*- coding: utf-8 -*-
# ===============================================================
#  author: hehuihui@zhiyitech.com
#  date: 2018/04/18 14:17
#  brief: 邮件发送器
# ===============================================================

import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr

import logging

logger = logging.getLogger(__name__)

SENDER = 'alert@zhiyitech.cn'     # 发件人邮箱账号
PASSWORD = 'BUpiEjgxJjNRKGAN'     # 发件人邮箱密码
RECEIVERS = ['data-infra@zhiyitech.cn']
#RECEIVERS = ['tagewalker@zhiyitech.cn']
MAIL_SERVER, MAIL_PORT = "smtp.exmail.qq.com", 465

def tolist(x):
    """
    将元素x转换为列表。
    :param x: any, 元素
    :return:
        - None, 如果x是空类型（None/0/[]/{}/""）
        - x, 如果x是list/tuple
        - [x], 其余情况
    """
    if not x:
        return None
    if isinstance(x, (tuple, list)):
        return x
    return [x]


class Mail(object):
    """
    Usage:
        >>> from util.mail import Mail
        >>> sender = Mail(receivers=['hehhpku@qq.com', 'huihui@zhiyitech.cn'])
        >>> sender.send_mail('title', 'content')
        >>> Mail().send_mail('title', 'content')
    """

    def __init__(self, nickname='data-infra', sender=SENDER, password=PASSWORD,
                 receivers=RECEIVERS, cc_receivers=None, logger=logger, mail_prefix=''):
        """
        :param nickname: str, 发件人昵称
        :param sender: str, 发件人邮箱账号
        :param password: str, 发件人邮箱密码
        :param receivers: str or list, 收件人邮箱账号
        :param logger: object, 日志对象
        :param mail_prefix: str, 邮件标题前缀
        """
        self.nickname = nickname
        self.sender = sender
        self.password = password
        self.receivers = tolist(receivers)
        self.cc_receivers = tolist(cc_receivers)
        self.logger = logger
        self.mail_prefix = mail_prefix

    def send_mail(self, subject, content, subtype='html', charset='utf-8'):
        """
        发送邮件
        :param subject: str, 邮件标题
        :param content: str, 邮件内容
        :param subtype: str, 邮件格式（例如：plain/html/json等）
        :param charset: str, 邮件字符编码
        :return: bool, 是否发送成功
        """
        try:
            msg = MIMEText(content, subtype, charset)
            msg['From'] = formataddr([self.nickname, self.sender])
            if self.receivers is not None:
                msg['To'] = ','.join(self.receivers)
            if self.cc_receivers is not None:
                msg['CC'] = ','.join(self.cc_receivers)
            msg['Subject'] = self.mail_prefix + subject

            # 企业邮箱发送邮件服务器 smtp.exmail.qq.com
            # qq邮箱发送邮件服务器  smtp.qq.com
            server = smtplib.SMTP_SSL(MAIL_SERVER, MAIL_PORT)
            server.login(self.sender, self.password)
            server.sendmail(self.sender, self.receivers, msg.as_string())
            server.quit()
        except Exception as e:
            self.logger.exception(e)
            return False
        return True

    def warn(self, subject, content):
        subject = '[WARN]' + subject
        self.send_mail(subject, content)

    def info(self, subject, content):
        subject = '[INFO]' + subject
        self.send_mail(subject, content)

    def error(self, subject, content):
        subject = '[ERROR]' + subject
        self.send_mail(subject, content)
