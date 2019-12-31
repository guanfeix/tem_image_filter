import time
import queue
import socket
import logging
import threading


from queue import Queue

from service.logging_service import logger

# logger = logging.getLogger('fuck')
logger.setLevel(logging.INFO)


free_que = {'queue': Queue()}
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)


class MyThread(threading.Thread):
    """
    一个模板线程类
    """
    def __init__(self, name, *args):
        super().__init__()
        self.name = name
        self.params = free_que
        self.run_state = False

    def __del__(self):
        # 仅提供 gc 前的 log，不影响delete 语句
        logger.info('{} destroy, release some resource'.format(self.name))

    def clone(self, *args):
        # 不能适应子类的clone,但让这种也比较少见程序有异常崩溃了
        return self._clone(self.name, *args)

    @classmethod
    def _clone(cls, name, *args):
        return cls(name, *args)

    def run(self):
        logger.info('thread {} start ...'.format(self.name))
        self.run_state = True
        self.work()
        logger.info('thread {} stop ...'.format(self.name))

    def stop(self):
        self.run_state = False
        print('over')
        logger.info('self.run_state: {}'.format(self.run_state))

    def get_state(self):
        return self.run_state

    def get_name(self):
        return self.name

    def work(self):
        q = self.params['queue']
        s.settimeout(5)
        while self.run_state:
            try:
                data = s.recv(8192)
                logger.info('thread {} working ...'.format(self.name))
                logger.info('self.run_state: {}'.format(self.run_state))
                msg = q.get(block=True, timeout=7)
                logger.info('get msg: {}'.format(msg))
                logger.info('self.run_state: {}'.format(self.run_state))
                time.sleep(1)
            except (queue.Empty, socket.timeout) as e:
                logger.exception('work fail: {}'.format(e))


if __name__ == '__main__':
    t1 = MyThread('shit')
    t1.start()
    free_que['queue'].put('shit')
    logger.warning('waring')
    print(threading.current_thread().getName())
    print(t1.getName())
    print(t1.is_alive())
    t1.stop()
    print(t1.is_alive())