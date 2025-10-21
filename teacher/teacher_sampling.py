import argparse, yaml, os
from pathlib import Path
import torch
from teacher.load_teacher import load_teacher_pipline
from tqdm import tqdm
import numpy as np
import torch
from data_utils.dataset import LatentDataset

def load_cfg(path):
    with open(path,'r') as f:
        import yaml
        return yaml.safe_load(f)
    
def generate_targets(cfg,teacher_ckpt=None):
    device=torch.device(cfg['training']['device'])
    model_id=cfg['teacher']['hf_model_id']
    pipeline=load_teacher_pipline(model_id=model_id,device=device)

    if teacher_ckpt:
        ck=torch.load(teacher_ckpt,map_location=device)
        if hasattr(pipeline, 'unet'):
            pipeline.unet.load_state_dict(ck['model'])
    
    out_dir=Path(cfg['data']['distill_target_dir'])
    out_dir.mkdir(parents=True,exist_ok=True)

    dataset=LatentDataset(cfg['data']['latent_dir'])
    for idx in range(len(dataset)):
        x0=dataset[idx]
        out=pipeline(
            num_inference_steps=cfg['teacher']['T'],
            generator=torch.Generator(device).manual_seed(42)
        )
        audio=out[0] if isinstance(out,(list,tuple)) else out
        np.savez_compressed(out_dir / f"target_{idx:06d}.npz", audio=audio)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--teacher_ckpt', default=None)
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    generate_targets(cfg, teacher_ckpt=args.teacher_ckpt)