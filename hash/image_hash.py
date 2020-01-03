from multiprocessing import cpu_count
import multiprocessing
import time
import os.path
from PIL import Image
import numpy
import scipy.fftpack

import numpy as np

class ImageHash():

    @staticmethod
    def _binary_array_to_hex(arr):
        bit_string = ''.join(str(b) for b in 1 * arr.flatten())
        width = int(numpy.ceil(len(bit_string)/4))
        return '{:0>{width}x}'.format(int(bit_string, 2), width=width)

    @staticmethod
    def average_hash(image, hash_size=8):
        if hash_size < 2:
            raise ValueError("Hash size must be greater than or equal to 2")

        image = image.convert("L").resize(
            (hash_size, hash_size), Image.ANTIALIAS)
        pixels = numpy.asarray(image)
        avg = pixels.mean()
        diff = pixels > avg
        return diff

    @staticmethod
    def phash(image, hash_size=8, highfreq_factor=4):
        if hash_size < 2:
            raise ValueError("Hash size must be greater than or equal to 2")
        img_size = hash_size * highfreq_factor
        image = image.convert("L").resize(
            (img_size, img_size), Image.ANTIALIAS)
        pixels = numpy.asarray(image)
        dct = scipy.fftpack.dct(scipy.fftpack.dct(pixels, axis=0), axis=1)
        dctlowfreq = dct[:hash_size, :hash_size]
        med = numpy.median(dctlowfreq)
        diff = dctlowfreq > med
        return diff

    @staticmethod
    def phash_simple(image, hash_size=8, highfreq_factor=4):
        img_size = hash_size * highfreq_factor
        image = image.convert("L").resize(
            (img_size, img_size), Image.ANTIALIAS)
        pixels = numpy.asarray(image)
        dct = scipy.fftpack.dct(pixels)
        dctlowfreq = dct[:hash_size, 1:hash_size+1]
        avg = dctlowfreq.mean()
        diff = dctlowfreq > avg
        return diff

    @staticmethod
    def dhash(image, hash_size=8):
        if hash_size < 2:
            raise ValueError("Hash size must be greater than or equal to 2")
        image = image.convert("L").resize(
            (hash_size + 1, hash_size), Image.ANTIALIAS)
        pixels = numpy.asarray(image)
        diff = pixels[:, 1:] > pixels[:, :-1]
        return diff

    @staticmethod
    def dhash_vertical(image, hash_size=8):
        image = image.convert("L").resize(
            (hash_size, hash_size + 1), Image.ANTIALIAS)
        pixels = numpy.asarray(image)
        diff = pixels[1:, :] > pixels[:-1, :]
        return diff

    @staticmethod
    def whash(image, hash_size=8, image_scale=None, mode='haar', remove_max_haar_ll=True):
        import pywt
        if image_scale is not None:
            assert image_scale & (
                image_scale - 1) == 0, "image_scale is not power of 2"
        else:
            image_natural_scale = 2**int(numpy.log2(min(image.size)))
            image_scale = max(image_natural_scale, hash_size)
        ll_max_level = int(numpy.log2(image_scale))
        level = int(numpy.log2(hash_size))
        assert hash_size & (hash_size-1) == 0, "hash_size is not power of 2"
        assert level <= ll_max_level, "hash_size in a wrong range"
        dwt_level = ll_max_level - level
        image = image.convert("L").resize(
            (image_scale, image_scale), Image.ANTIALIAS)
        pixels = numpy.asarray(image) / 255

        if remove_max_haar_ll:
            coeffs = pywt.wavedec2(pixels, 'haar', level=ll_max_level)
            coeffs = list(coeffs)
            coeffs[0] *= 0
            pixels = pywt.waverec2(coeffs, 'haar')
        coeffs = pywt.wavedec2(pixels, mode, level=dwt_level)
        dwt_low = coeffs[0]
        med = numpy.median(dwt_low)
        diff = dwt_low > med
        return diff
