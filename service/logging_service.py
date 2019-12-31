import logging

# FORMAT = '%(levelname)s %(asctime)s %(threadName)s %(filename)s %(funcName)s %(lineno)d: %(message)s'
FORMAT = '%(levelname)s %(asctime)s [%(thread)d] %(filename)s %(funcName)s %(lineno)d: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)
