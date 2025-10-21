#!/usr/bin/env python3
import argparse, yaml
from distill.progressive_distill import train_student_stage
def load_cfg(path):
    import yaml
    with open(path,'r') as f: return yaml.safe_load(f)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    train_student_stage(cfg, student_steps=cfg['distillation']['final_steps'])
