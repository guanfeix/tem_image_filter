
import logging
import time
import numpy as np
from dynaconf import settings
from util.redis_db import RedisClient
from util.common_util import CommonUtil

logger = logging.getLogger(__name__)


class IndexBucket(object):
    def __init__(self, prefix=''):
        self.prefix = prefix
        self.redis_client = RedisClient(**settings.DEDUP_TEM_INCRE_REDIS_CONFIG)
        logger.info('redis_client %s', self.redis_client.redis_db)

    def get_value(self, key):
        logger.debug('get value for key: {}'.format(key))
        k = self.prefix + key
        value = self.redis_client.set_members(k)
        logger.debug('key: {}, size: {}'.format(key, len(value)))
        return value

    def add_value(self, key, value):
        logger.debug('add value for key: {}'.format(key))
        k = self.prefix + key
        return self.redis_client.set_add(k, value)

    def remove_value(self, key, value):
        logger.debug('remove value for key: {}'.format(key))
        k = self.prefix + key
        return self.redis_client.set_remove(self.k, value)


class IndexBucketDict(object):
    def __init__(self, prefix=''):
        self.prefix = prefix
        self.dict_client = dict()

    def get_value(self, key):
        k = self.prefix + key
        value = self.dict_client.get(key, set())
        return value

    def add_value(self, key, value):
        k = self.prefix + key
        v_set = self.get_value(key)
        v_set.add(value)
        self.dict_client[key] = v_set
        #logger.info('dict_client: {}'.format(self.dict_client))
        return v_set

    def remove_value(self, key, value):
        k = self.prefix + key
        return self.get_value(key).remove(value)

class SimHashIndex(object):

    def __init__(self, bits=64, neark=2, 
        insert_enable=bool(settings.FEATURE_INSERT),
        prefix='', redis_bucket=True):
        self.bits = bits
        self.neark = neark
        self.insert_enable = insert_enable
        self.segments = self.neark + 1
        self.offsets = [self.bits // (self.segments) * i for i in range(self.segments)]
        self.bucket = IndexBucket(prefix) if redis_bucket else IndexBucketDict(prefix)

        logger.info('feature insert: {}'.format(self.insert_enable))

    def compute_keys_a(self, hash):
        for i, offset in enumerate(self.offsets):
            if i == (len(self.offsets) - 1):
                m = 2 ** (self.bits - offset) - 1
            else:
                m = 2 ** (self.offsets[i + 1] - offset) - 1
            c = hash >> offset & m
            yield '%x:%x' % (c, i)

    def compute_keys(self, feature):
        feature_bin = list('{0:064b}'.format(feature))
        bits_list = {}
        for i in range(self.segments):
            bits_list[i] = []

        for i, b in enumerate(feature_bin):
            k = i % self.segments
            bits_list[k].append(b)
        
        for k in bits_list.keys():
            key = '{}:{}'.format('{:x}'.format(int(''.join(bits_list[k]), 2)), k)
            yield key

    def get_value(self, hash):
        value = set()
        for key in self.compute_keys(hash):
            values = self.bucket.get_value(key)
            '''
            if values and len(values) > 0:
                break
            '''
            for v in values:
                value.add(v)

        return value

    def add_value(self, hash, value):

        for key in self.compute_keys(hash):
            self.bucket.add_value(key, value)

    def remove_value(self, hash, value):
        for key in self.compute_keys(hash):
            self.bucket.remove_value(key, value)

    def check_if_exist(self, hash, value):
        dups = self.get_near_dups(hash)
        if len(dups) > 0:
            logger.debug('duplicate features len: {}'.format(len(dups)))
            return dups

        if self.insert_enable:
            self.add_value(hash, value)
        return None

    def get_near_dups_new(self, hash):

        dups = self.get_value(hash)
        return self.compute_features_distance_split(hash, list(dups))

    def get_near_dups(self, hash):

        dups = self.get_value(hash)
        ans = set()
        for dup in set(dups):
            dup = int(dup)
            d = self.compute_distance(hash, dup)
            if d <= self.neark:
                #s = '{0:064b} <-> {0:064b}'.format(hash, dup)
                #logger.info('{} distance: {}'.format(s, d))
                ans.add(dup)
        return list(ans)

    def compute_distance(self, hash1, hash2):
        x = (hash1 ^ hash2) & ((1 << self.bits) - 1)
        ans = 0
        while x and self.neark >= ans:

            ans += (1 & x)
            x = (x >> 1)
        
        if ans <= self.neark:
            logger.debug(
                'compute distance between {} <-> {}: {}'.format(hash1, hash2, ans))
        return ans

    def compute_distance_1(self, hash1, hash2):
        x = (hash1 ^ hash2) & ((1 << self.bits) - 1)
        ans = 0
        while x:
            ans += 1
            x &= x - 1
        logger.debug(
            'compute distance between {} <-> {}: {}'.format(hash1, hash2, ans))
        return ans

    def compute_features_distance(self, query_feature, feature_lib, distance = 3):

        start_time = time.time()
        distance_list = np.sum(pow(query_feature - feature_lib, 2), axis=1)

        distance_list = np.where(distance_list < distance)
        # logger.info('distance list: {}, cost time: {}'.format(distance_list, (time.time() - start_time)))
        return distance_list

    def compute_features_distance_split(self, query_feature, feature_lib, distance = 3):

        start_time = time.time()
        query_array = np.array(self.bin_array(query_feature, 64))
        split_feature_list = CommonUtil.split_list(feature_lib, 64)
        distance_list = []
        for feature_list in split_feature_list:
            array_list = []
            for f in feature_list:
                array_list.append(self.bin_array(int(f), 64))

            lib_array = np.array(array_list)
            distance_list = self.compute_features_distance(query_array, lib_array)[0]
            if (distance_list.size > 0):
                break
        
        duplicate_features = []
        for d in distance_list:
            duplicate_features.append(feature_list[d])
        # logger.info('sort distance len:{}, cost time: {}'.format(distance_list.size, (time.time() - start_time)))
        return duplicate_features

    @classmethod
    def bin_array(self, feature, bits=64):
        feature_bits = []
        for i in range(bits):
            feature_bits.append(1 & feature)
            feature >>= 1
        return feature_bits

    @classmethod
    def bin_64bits(self, feature):
        feature_bits = list('{0:064b}'.format(feature))
        return feature_bits