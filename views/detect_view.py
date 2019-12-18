from flask import Flask
from flask import request
from flask import make_response, jsonify
from flask.views import View

import traceback
import sys
import time
import json
import logging

logger = logging.getLogger(__name__)

from versioninfo import VersionInfo

import socket
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
    def init_version(self):
        self.version = VersionInfo()

    @classmethod
    def init_reporter(self, reporter):
        self.reporter = reporter

    @classmethod
    def init_filter(self, image_filter):
        self.image_filter = image_filter

    @classmethod
    def init_color_detect(self, color_detect):
        self.color_detect = color_detect

    @classmethod
    def init_color_recognition(self, color_recognition):
        self.color_recognition = color_recognition

    @classmethod
    def init_detect(self, category_detect):
        self.category_detect = category_detect

    @classmethod
    def init_all_detect(self, category_all_detect):
        self.category_all_detect = category_all_detect

    @classmethod
    def init_recognition(self, category_recognition):
        self.category_recognition = category_recognition

    @classmethod
    def init_clothes_recognition(self, clothes_recognition):
        self.clothes_recognition = clothes_recognition

    @classmethod
    def init_clothes_region(self, clothes_region):
        self.clothes_region = clothes_region

    @classmethod
    def init_clothes_technology(self, clothes_technology):
        self.clothes_technology = clothes_technology

    @classmethod
    def init_clothes_texture(self, clothes_texture):
        self.clothes_texture = clothes_texture

    @classmethod
    def init_clothes_style(self, clothes_style):
        self.clothes_style = clothes_style

    @classmethod
    def init_clothes_fabric(self, clothes_fabric):
        self.clothes_fabric = clothes_fabric

    @classmethod
    def init_clothes_type(self, clothes_type):
        self.clothes_type = clothes_type

    @classmethod
    def init_clothes_material(self, clothes_material):
        self.clothes_material = clothes_material

    @classmethod
    def init_score(self, image_score):
        self.image_score = image_score

    @classmethod
    def init_synthesize_recognition(self, synthesize_recognition):
        self.synthesize_recognition = synthesize_recognition

    def dispatch_request(self):
        pass