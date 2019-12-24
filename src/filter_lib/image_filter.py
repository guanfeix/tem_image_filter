import os
import sys
path = os.path.realpath(__file__)
pdir = os.path.dirname(path)
for i in range(2):
    pdir = os.path.dirname(pdir)

import logging
from .image_filter_lib import ImageFilter

logger = logging.getLogger(__name__)

pdir = pdir + '/models'
models_path = {
    "face_detect_model_path": pdir + "/detect_model",
    "clothes_detect_model_path": pdir + "/eland_detect_v3.h5",
    "complexion_model_path": pdir + "/complexion_model_v1.h5",
    'text_model_path': pdir + '/text_classify_v1.h5',
}


class ImageFilterConstructor(object):
    def __init__(self):
        pass

    @classmethod
    def instance(cls):
        return ImageFilter(**models_path)
