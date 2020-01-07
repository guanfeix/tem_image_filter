import socket

from flask.views import View

from versioninfo import VersionInfo

hostname = socket.gethostname()


class DetectView(View):

    methods = ['POST']
    image_filter = None
    color_detect = None
    category_detect = None
    category_all_detect = None
    category_recognition = None
    clothes_recognition = None
    clothes_region = None
    clothes_technology = None
    clothes_texture = None
    clothes_style = None
    clothes_fabric = None
    clothes_type = None
    clothes_material = None
    image_score = None
    synthesize_recognition = None

    reporter = None

    @classmethod
    def init_version(cls):
        cls.version = VersionInfo()

    @classmethod
    def init_reporter(cls, reporter):
        cls.reporter = reporter

    @classmethod
    def init_filter(cls, image_filter):
        cls.image_filter = image_filter

    @classmethod
    def init_color_detect(cls, color_detect):
        cls.color_detect = color_detect

    @classmethod
    def init_color_recognition(cls, color_recognition):
        cls.color_recognition = color_recognition

    @classmethod
    def init_detect(cls, category_detect):
        cls.category_detect = category_detect

    @classmethod
    def init_all_detect(cls, category_all_detect):
        cls.category_all_detect = category_all_detect

    @classmethod
    def init_recognition(cls, category_recognition):
        cls.category_recognition = category_recognition

    @classmethod
    def init_clothes_recognition(cls, clothes_recognition):
        cls.clothes_recognition = clothes_recognition

    @classmethod
    def init_clothes_region(cls, clothes_region):
        cls.clothes_region = clothes_region

    @classmethod
    def init_clothes_technology(cls, clothes_technology):
        cls.clothes_technology = clothes_technology

    @classmethod
    def init_clothes_texture(cls, clothes_texture):
        cls.clothes_texture = clothes_texture

    @classmethod
    def init_clothes_style(cls, clothes_style):
        cls.clothes_style = clothes_style

    @classmethod
    def init_clothes_fabric(cls, clothes_fabric):
        cls.clothes_fabric = clothes_fabric

    @classmethod
    def init_clothes_type(cls, clothes_type):
        cls.clothes_type = clothes_type

    @classmethod
    def init_clothes_material(cls, clothes_material):
        cls.clothes_material = clothes_material

    @classmethod
    def init_score(cls, image_score):
        cls.image_score = image_score

    @classmethod
    def init_synthesize_recognition(cls, synthesize_recognition):
        cls.synthesize_recognition = synthesize_recognition
