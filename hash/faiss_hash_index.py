import logging

try:
    import faiss
except:
    pass
    
logger = logging.getLogger(__name__)


class FaissHashIndex(object):
    def __init__(self, d=64, index_path='/workspace/zhiyi/data/faiss.index'):
        self.faiss_sub_index = faiss.IndexFlatL2(d)
        self.faiss_index = faiss.IndexIDMap2(self.faiss_sub_index)
        self.index_path = index_path

    def get_index(self):
        return self.faiss_index

    def set_index(self, index):
        self.faiss_index = index

    def is_trained(self):
        return self.faiss_index.is_trained

    def get_ntotal(self):
        return self.faiss_index.ntotal

    def add_data(self, data):
        self.faiss_index.add(data)

    def add_data_with_ids(self, data, ids):
        self.faiss_index.add_with_ids(data, ids)

    def search(self, q, k=10):
        return self.faiss_index.search(q, k)

    def save_index(self):
        try:
            logger.info('save faiss index to: {}'.format(self.index_path))
            faiss.write_index(self.faiss_index, self.index_path)
            return True
        except Exception as e:
            logger.error('save index fail: {}'.format(e))
        return False

    def load_index(self):
        try:
            faiss_index = faiss.read_index(
                self.index_path, faiss.IO_FLAG_MMAP)
            logger.info('load index(ntotal: {}, is_trained: {}) from {}'.format(
                faiss_index.ntotal, faiss_index.is_trained, self.index_path))
            return faiss_index
        except Exception as e:
            logger.error('load index fail: %s' % e)
        return None
