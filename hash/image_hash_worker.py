import logging

import numpy as np

from PIL import Image
from io import BytesIO

from .image_hash import ImageHash

logger = logging.getLogger(__name__)


class ImageHashWorker(object):

    def __init__(self, hash_type='phash'):
        self.hash_type = hash_type
        self.image_hash = ImageHash()

    def hash_pil_image(self, image):

        if not image:
            return None

        image_hash = self.image_hash
        hash_type = self.hash_type
        if hash_type == "phash":
            img_hash = image_hash.phash(image, hash_size=8, highfreq_factor=1)
        elif hash_type == "ahash":
            hash_size = 6
            img_hash = image_hash.average_hash(image, hash_size=6)
        elif hash_type == "dhash":
            img_hash = image_hash.dhash(image, hash_size=10)
        elif hash_type == "whash":
            img_hash = image_hash.whash(
                image, image_scale=64, hash_size=8, mode="db4")

        hash_sum = int(np.sum([2**(64-(index+1))*int(value)
                               for index, value in enumerate(img_hash.flatten())]))
        return hash_sum

    def filter_hash_image(self, url, data):
        image = Image.open(BytesIO(data))
        return [url, self.hash_pil_image(image)]






