from abc import ABC, abstractmethod

class AttentionMask(ABC):
    @abstractmethod
    def __call__(self, attn, **kwargs):
        ...