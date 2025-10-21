#!/usr/bin/env python3
import argparse, yaml, os
from pathlib import Path
import torch
import numpy as np
import soundfile as sf
from models.codec_wrapper import CodecWrapper
from student.student_model import SmallUNet1D

def load_cfg(path):
    with open(path,'r') as f:
        import yaml
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--student_ckpt', required=True)
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    device = torch.device(cfg['training']['device'])
    codec = CodecWrapper(sr=cfg['data']['sample_rate'], bitrate=cfg['encodec']['bitrate'])
    # load student
    student = SmallUNet1D(in_ch=4, base_ch=max(32,int(256*cfg['distillation']['student_scale']))).to(device)
    ck = torch.load(args.student_ckpt, map_location=device)
    student.load_state_dict(ck['model'])
    student.eval()
    out_dir = Path(cfg['logging']['samples_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    # produce N samples by random or by using dataset seeds
    for i in range(10):
        z = torch.randn(1,4,1600).to(device)  # 1600 frames is a placeholder - align with your latent frame rate
        with torch.no_grad():
            pred = student(z, torch.ones(1, device=device))
            pred_np = pred.cpu().numpy()[0]
            try:
                wav = codec.decode_codes(pred_np)
                out_path = out_dir / f"sample_{i:03d}.wav"
                sf.write(str(out_path), wav.squeeze(), cfg['data']['sample_rate'])
                print("Wrote", out_path)
            except Exception as e:
                print("decode failed:", e)
    print("For FAD: see Google FAD repo. For CLAP embeddings: load CLAP model and compute embeddings on output folder.")
