from .base import AttentionBackend
from ..registries import register_backend

@register_backend("efficient")
class EfficientBackend(AttentionBackend):
    def __call__(self, q, k, v, **kwargs):
        rhok = k.softmax(dim=-2)
        rhoq = q.softmax(dim=-1)
        rhokv = rhok.transpose(-2, -1) @ v
        return rhoq @ rhokv