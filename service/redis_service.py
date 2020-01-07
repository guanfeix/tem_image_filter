import os
import logging
import redis
import fcntl
import socket
import struct
import psutil
import platform

from urllib import parse
# OSSURL_INTERNAL = os.getenv('OSSURL_INTERNAL', True)
OSSURL_INTERNAL = os.getenv('OSSURL_INTERNAL', False)

port = 0
password = None
if socket.gethostbyname(socket.gethostname()) in ['127.0.0.1', '192.168.1.51']:
    REDIS_URL = 'redis://127.0.0.1:6379'
else:
    REDIS_URL = 'redis://gpu003:6379'
    password = 'zhiyiredis'
logger = logging.getLogger(__name__)
logger.info('REDIS_URL: %s', REDIS_URL)


def __get_server_ip():
    """
    获取当前服务器ip地址,mac 不可用
    :return:
    """
    try:
        if platform.system() == 'Windows':
            ip = socket.gethostbyname(socket.getfqdn(socket.gethostname()))
        elif platform.system() == 'Linux':  # 获取eth0网卡地址
            def get_ip_address(ifname):
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                return socket.inet_ntoa(fcntl.ioctl(
                    s.fileno(),
                    0x8915,  # SIOCGIFADDR
                    struct.pack('256s', bytes(ifname[:15], 'utf-8'))
                )[20:24])
            try:
                ip = get_ip_address('eno1')
            except Exception as e:
                ip = get_ip_address('eth0')
        else:
            ip = '127.0.0.1'
        return ip
    except Exception as err:
        print(err)
        return ''


def get_running_port():
    """
    减少计算，公共变量缓存
    :return:
    """
    global port
    if port == 0:
        try:
            pid = os.getpid()
            ppid= os.getppid()
            print('\n\r')
            print(pid, ppid)
            p = psutil.Process(pid)

            port = [i.laddr.port for i in p.connections() if i.status=='LISTEN'][0]
            print(port, p.connections())
        except Exception as e:
            p = psutil.Process(ppid)
            print(p.connections())
            logging.exception('Error: %s', e)
            port = 0
    return port


host_name = socket.gethostname()

# port = __get_running_port()

server_ip = __get_server_ip()


class RedisService(redis.Redis):
    """
    继承了redis.Redis的RedisService，初始化可以得到redis_client，不需要重复输入host，port，password等信息，方便统一管理
    """
    # redis_url = getattr(settings, "REDIS_URL", "")
    redis_split_result = parse.urlsplit(REDIS_URL)
    redis_host = redis_split_result.hostname
    redis_port = redis_split_result.port

    def __init__(self):
        redis.Redis.__init__(self, host=self.redis_host, port=self.redis_port,
                             password=password, decode_responses=True)
        # super().__init__(host=self.redis_host, port=self.redis_port, decode_responses=True)
        # super(RedisService, self).__init__(host=self.redis_host, port=self.redis_port, decode_responses=True)


redis_client = RedisService()
