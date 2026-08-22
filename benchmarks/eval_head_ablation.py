import argparse
import torch
import yaml
import numpy as np
import os
from utils.datasets import build_dataset
from engine_finetune import evaluate
import models_vit
from vision_transformer import Attention
from main_finetune import _apply_attention_cfg_to_model

def get_args_parser():
    parser = argparse.ArgumentParser('Head Ablation Evaluation', add_help=False)
    parser.add_argument('--dataset', default='c10', type=str)
    parser.add_argument('--model', default='vit_base_patch16', type=str)
    parser.add_argument('--finetune', default='', required=True, help='Path to checkpoint')
    parser.add_argument('--attn-cfg', type=str, default='configs/attention/vit_fibottention.yaml')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--nb_classes', default=10, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true', default=True)
    return parser

def main(args):
    device = torch.device(args.device)
    dataset_val = build_dataset(is_train=False, args=args)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False
    )

    model = models_vit.__dict__[args.model](num_classes=args.nb_classes, global_pool=True)
    
    with open(args.attn_cfg, "r") as f:
        attn_cfg = yaml.safe_load(f).get("attention")
    _apply_attention_cfg_to_model(model, attn_cfg)

    checkpoint = torch.load(args.finetune, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    model.to(device)
    model.eval()

    print("\n[Base Evaluation]")
    base_stats = evaluate(data_loader_val, model, device)
    base_acc = base_stats['acc1']
    print(f"Base Accuracy: {base_acc:.2f}%\n")

    num_heads = model.blocks[0].attn.num_heads
    head_dim = model.blocks[0].attn.head_dim
    drops = []

    for h in range(num_heads):
        print(f"--- Ablating Head {h+1}/{num_heads} ---")
        
        orig_weights = {}
        for name, m in model.named_modules():
            if isinstance(m, Attention):
                orig_weights[name] = m.proj.weight.data.clone()
                m.proj.weight.data[:, h*head_dim : (h+1)*head_dim] = 0.0

        stats = evaluate(data_loader_val, model, device)
        drop = base_acc - stats['acc1']
        drops.append(drop)
        print(f"Accuracy with Head {h} ablated: {stats['acc1']:.2f}% (Drop: {drop:.2f}%)")

        for name, m in model.named_modules():
            if isinstance(m, Attention):
                m.proj.weight.data.copy_(orig_weights[name])

    print("\n=== Ablation Summary ===")
    print(f"Average Accuracy Drop per Ablated Head: {np.mean(drops):.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Head Ablation Evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)