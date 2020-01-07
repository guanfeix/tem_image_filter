from dynaconf import settings
from src.filter_lib.image_filter_lib import ImageFilter


class ImageFilterConstructor(object):
    """
    抽象和具体分离，抽象不依赖与具体实现,不多余参数复杂且多变有必要，再抽象出来初始化模块,再抽象一下配置未见和逻辑代码分离
    """
    def __init__(self):
        pass

    @classmethod
    def instance(cls):
        return ImageFilter(**settings.MODELS_PATH)
