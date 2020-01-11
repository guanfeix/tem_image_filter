import os

SERVICE_LIST = ['ImageFilter', 'CategoryDetect', 'CategoryRecognition']

# 模型路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODELS_PATH = {
    "face_detect_model_path": os.path.join(MODEL_DIR, "detect_model"),
    "clothes_detect_model_path": os.path.join(MODEL_DIR, "eland_detect_v3.h5"),
    "complexion_model_path": os.path.join(MODEL_DIR, "complexion_model_v1.h5"),
    'text_model_path': os.path.join(MODEL_DIR, 'text_classify_v1.h5'),
}

# 去重
FEATURE_INSERT = os.getenv('FEATURE_INSERT', '1')
DEDUPLICATE = True

