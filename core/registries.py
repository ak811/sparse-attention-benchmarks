_BACKENDS = {}
_MASKS = {}

def register_backend(name):
    def deco(fn):
        _BACKENDS[name.lower()] = fn
        return fn
    return deco

def get_backend(name):
        return _BACKENDS[name.lower()]

def register_mask(name):
    def deco(fn):
        _MASKS[name.lower()] = fn
        return fn
    return deco

def get_mask(name):
    return _MASKS[name.lower()]
