from .registries import get_backend, get_mask

class AttentionDispatcher:
    def __init__(self, backend_name, backend_kwargs=None, mask_name=None, mask_kwargs=None):
        backend_ctor = get_backend(backend_name)
        self.backend = backend_ctor(**(backend_kwargs or {})) if callable(backend_ctor) else backend_ctor
        self.mask_fn = get_mask(mask_name)(**(mask_kwargs or {})) if mask_name else None

    def __call__(self, q, k, v, **kwargs):
        return self.backend(q, k, v, mask_fn=self.mask_fn, **kwargs)
