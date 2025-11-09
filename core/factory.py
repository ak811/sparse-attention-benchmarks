# image_classification/core/factory.py
from .attention import AttentionDispatcher

# >>> add these blocks so decorators run once <<<
# Backends
from .backends import vit as _b_vit                 # registers "vit"
from .backends import efficient as _b_eff           # registers "efficient"
from .backends import linformer as _b_lin           # registers "linformer"
from .backends import vit_qchunk as _b_vit_qchunk   # registers "vit_qchunk"
from .backends import vit_qchunk_topk as _b_vit_topk  # registers "vit_topk"
from .backends import efficient_local as _b_eff_local

# Masks
from .masks import none as _m_none                  # registers "none"
from .masks import sparse as _m_sparse              # registers "sparse"
from .masks import fibottention as _m_fibo          # registers "fibottention"
from .masks import fibottention_crossdiag as _m_fibo_xdiag  # registers "fibottentioncrossdiag"
from .masks import topk as _m_topk
from .masks import random as _m_random
from .masks import bigbird as _m_bigbird            # registers "bigbird"
from .masks import longformer as _m_longformer
# <<< end add >>>

from .backends.linformer import LinformerProjector

def build_attention_from_cfg(cfg):
    bname = cfg.get("backend", "vit")
    if bname == "linformer":
        projector = LinformerProjector(**cfg.get("backend_kwargs", {}).get("projector", {}))
        return AttentionDispatcher(
            "linformer", {"projector": projector},
            cfg.get("mask"), cfg.get("mask_kwargs", {})
        )
    return AttentionDispatcher(
        bname, cfg.get("backend_kwargs", {}),
        cfg.get("mask"), cfg.get("mask_kwargs", {})
    )
