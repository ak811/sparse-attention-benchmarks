from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Optional, Callable

class AttentionBackend(ABC):
    @abstractmethod
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        estep=None,
        N: Optional[int] = None,
        num_heads: Optional[int] = None,
        depth_id: Optional[int] = None,
        device: Optional[torch.device] = None,
        attn_drop: Optional[nn.Dropout] = None,
        mask_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        save_hook: Optional[Callable[[torch.Tensor], None]] = None,
    ) -> torch.Tensor:
        ...