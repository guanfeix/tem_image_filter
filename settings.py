import os
import copy

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

# 去重模块
# DEDUP_TEM_INCRE_REDIS_CONFIG = REDIS_CONFIG_TEST
DEDUP_TEM_INCRE_REDIS_CONFIG = copy.deepcopy(REDIS_CONFIG)
DEDUP_TEM_INCRE_REDIS_CONFIG['db'] = 3
FEATURE_INSERT = os.getenv('FEATURE_INSERT', '1')
DEDUPLICATE = True

# 模型路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODELS_PATH = {
    "face_detect_model_path": os.path.join(MODEL_DIR, "detect_model"),
    "clothes_detect_model_path": os.path.join(MODEL_DIR, "eland_detect_v3.h5"),
    "complexion_model_path": os.path.join(MODEL_DIR, "complexion_model_v1.h5"),
    'text_model_path': os.path.join(MODEL_DIR, 'text_classify_v1.h5'),
}

# 是否内网
OSSURL_INTERNAL = os.getenv('OSSURL_INTERNAL', True)
# OSSURL_INTERNAL = os.getenv('OSSURL_INTERNAL', False)
