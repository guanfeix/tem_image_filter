import redis

import logging
logger = logging.getLogger(__name__)


class RedisClient(object):
    def __init__(self, host='localhost', port=6379, db=0, password=None):
        try:
            self.redis_db = redis.StrictRedis(host=host, port=port, db=db, password=password, socket_keepalive=True)
            self.redis_db.ping()
        except Exception as e:
            logger.error('exception: {}'.format(e))
            raise Exception('connect redis db fail: {}'.format(e))

    def get(self, key):
        value = self.redis_db.get(key)
        logger.debug('get for key: {}, value: {}'.format(key, value))
        return value

    def set(self, key, value, ex=None):
        res = self.redis_db.set(key, value, ex)
        logger.debug('set for key: {}, value: {}, res: {}'.format(key, value, res))
        return res

    def incrby(self, key, amount):
        res = self.redis_db.incrby(key, amount)
        logger.debug('incrby for key: {}, amount: {}, res: {}'.format(key, amount, res))
        return res

    def list_len(self, key):
        len = self.redis_db.llen(key) 
        logger.debug('llen list for key: {}, len: {}'.format(key, len))
        return len

    def list_push(self, key, value):   
        len = self.redis_db.lpush(key, value) 
        logger.debug('push list for key: {}, value: {}, len: {}'.format(key, value, len))   
        return len

    def list_pull(self, key, start=0, end=-1):
        values = self.redis_db.lrange(key, start, end)
        logger.debug('pull list for key: {}, values: {}'.format(key, values))
        return values

    def list_pop(self, key):
        value = self.redis_db.lpop(key)
        logger.debug('pop list for key: {}, value: {}'.format(key, value))
        return value

    def list_blpop(self, key):
        value = self.redis_db.blpop(key)
        logger.debug('block pop list for key: {}, value: {}'.format(key, value))
        return value

    def list_remove(self, key, value):
        result = self.redis_db.lrem(key, 1, value)
        logger.debug('remove list value for key: {}, value: {}, result: {}'.format(key, value, result))
        return result

    def set_len(self, key):
        len = self.redis_db.scard(key)
        logger.debug('len set for key: {}, len: {}'.format(key, len))   
        return len

    def set_add(self, key, value):   
        len = self.redis_db.sadd(key, value) 
        logger.debug('add set for key: {}, value: {}, len: {}'.format(key, value, len))   
        return len

    def set_members(self, key):
        value = self.redis_db.smembers(key)
        logger.debug('pull set members for key: {}, values: {}'.format(key, value))
        return value

    def set_remove(self, key, value):
        result = self.redis_db.srem(key, value)
        logger.debug('remove set value for key: {}, value: {}, result: {}'.format(key, value, result))
        return result

    def keys(self, pat='*'):
        result = self.redis_db.keys(pat)
        logger.debug('get keys size: {}'.format(len(result)))
        return result

    def delete(self, key):
        result = self.redis_db.delete(key)
        logger.debug('delete key {}: {}'.format(key, result))
        return result