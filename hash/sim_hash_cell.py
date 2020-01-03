import hashlib

def _hashfunc(x):
    return int(hashlib.md5(x).hexdigest(), 16)

class SimHashCell(object):
    def __init__(
        self, value, bits=64, reg=r'[\w\u4e00-\u9fcc]+', hashfunc=_hashfunc):
        self.bits = bits
        self.reg = reg
        self.value = None
        self.hashfunc = hashfunc

        if isinstance(value, SimHashCell):
            self.value = value.value
        elif isinstance(value, str):
            self.value = imghash.trans_img_hash(value)   
        elif isinstance(value, collections.Iterable):
            self.build_by_features(value)
        elif isinstance(value, numbers.Integral):
            self.value = value
        else:
            raise Exception('Bad parameter with type {}'.format(type(value)))

    def _slide(self, content, width=4):
        return [content[i:i + width] for i in range(max(len(content) - width + 1, 1))]

    def _tokenize(self, content):
        content = content.lower()
        content = ''.join(re.findall(self.reg, content))
        ans = self._slide(content)
        return ans

    def build_by_text(self, content):
        features = self._tokenize(content)
        features = {k:sum(1 for _ in g) for k, g in groupby(sorted(features))}
        return self.build_by_features(features)

    def build_by_features(self, features):
        v = [0] * self.f
        masks = [1 << i for i in range(self.f)]
        if isinstance(features, dict):
            features = list(features.items())
        for f in features:
            if isinstance(f, str):
                h = self.hashfunc(f.encode('utf-8'))
                w = 1
            else:
                assert isinstance(f, collections.Iterable)
                h = self.hashfunc(f[0].encode('utf-8'))
                w = f[1]
            for i in range(self.f):
                v[i] += w if h & masks[i] else -w
        ans = 0
        for i in range(self.f):
            if v[i] > 0:
                ans |= masks[i]
        self.value = ans

    def distance(self, another):
        assert self.f == another.f
        x = (self.value ^ another.value) & ((1 << self.f) - 1)
        ans = 0
        while x:
            ans += 1
            x &= x - 1
        return ans