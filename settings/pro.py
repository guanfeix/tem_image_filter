from settings.common import *
import copy
REDIS_CONFIG = {
    'db': os.getenv('redis_db', '0'),
    'host': os.getenv('REDIS_HOST', 'gpu003'),
    'port': int(os.getenv('REDIS_PORT', '6379')),
    'password': os.getenv('REDIS_PASSWD', 'zhiyiredis'),
}
# 去重模块
DEDUP_TEM_INCRE_REDIS_CONFIG = copy.deepcopy(REDIS_CONFIG)
DEDUP_TEM_INCRE_REDIS_CONFIG['db'] = 2

# 是否内网
OSSURL_INTERNAL = os.getenv('OSSURL_INTERNAL', True)
