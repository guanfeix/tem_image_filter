import os
import copy
INCLUDES_FOR_DYNACONF = "settings/settings.py"
SERVICE_LIST = ['ImageFilter', 'CategoryDetect', 'CategoryRecognition']

REDIS_CONFIG_TEST = {
    'db': os.getenv('redis_db', '0'),
    'host': os.getenv('REDIS_HOST', 'localhost'),
    'port': int(os.getenv('REDIS_PORT', '6379')),
    'password': os.getenv('REDIS_PASSWD', ''),
}

REDIS_CONFIG = {
    'db': os.getenv('redis_db', '0'),
    'host': os.getenv('REDIS_HOST', 'gpu003'),
    'port': int(os.getenv('REDIS_PORT', '6379')),
    'password': os.getenv('REDIS_PASSWD', 'zhiyiredis'),
}
# REDIS_CONFIG = REDIS_CONFIG_TEST


# DEDUP_TEM_INCRE_REDIS_CONFIG = REDIS_CONFIG_TEST
DEDUP_TEM_INCRE_REDIS_CONFIG = copy.deepcopy(REDIS_CONFIG)
DEDUP_TEM_INCRE_REDIS_CONFIG['db'] = 2
FEATURE_INSERT = os.getenv('FEATURE_INSERT', '1')

DEDUPLICATE = True
