from .attention import AttentionDispatcher

# Backends
from .backends import vit as _b_vit                  
from .backends import efficient as _b_eff            
from .backends import linformer as _b_lin            
from .backends import vit_qchunk as _b_vit_qchunk    
from .backends import vit_qchunk_topk as _b_vit_topk 
from .backends import efficient_local as _b_eff_local
from .backends import fibottention_sparse as _b_fibo_sparse 

# Masks
from .masks import none as _m_none                  
from .masks import sparse as _m_sparse              
from .masks import fibottention as _m_fibo          
from .masks import fibottention_crossdiag as _m_fibo_xdiag  
from .masks import topk as _m_topk
from .masks import random as _m_random
from .masks import bigbird as _m_bigbird            
from .masks import longformer as _m_longformer

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