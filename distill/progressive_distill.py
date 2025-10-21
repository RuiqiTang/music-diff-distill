import argparse, yaml, os
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from student.student_model import SmallUNet1D
from data_utils.dataset import LatentDataset
from tqdm import tqdm
import torch.optim as optim

def load_cfg(path):
    with open(path,'r') as f:
        import yaml
        return yaml.safe_load(f)
    
def train_student_stage(cfg,student_steps,init_ckpt=None,target_dir=None):
    device=torch.device(cfg['training']['device'])
    target_dir=target_dir or cfg['data']['distill_target_dir']
    dataset=LatentDataset(target_dir)
    dataloader=DataLoader(
        dataset,
        batch_size=cfg['distillation'].get('batch_size',cfg['teacher']['batch_size']),
        shuffle=True,
        num_workers=cfg['training']['num_workers'],
        pin_memory=True
    )
    in_ch=4
    base_ch=max(32,int(256*cfg['distillation']['student_scale']))
    student=SmallUNet1D(in_ch=in_ch,base_ch=base_ch).to(device)
    if init_ckpt:
        ck=torch.load(init_ckpt,map_location=device)
        student.load_state_dict(ck['model'])
    opt=optim.AdamW(student.parameters(),lr=cfg['distillation']['student_lr'])
    mse=torch.nn.MSELoss()

    global_step=0
    ckpt_dir=Path(cfg['logging']['ckpt_dir'])
    ckpt_dir.mkdir(parents=True,exist_ok=True)
    epochs=cfg['distillation'].get('epochs_per_stage',2000)

    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"student_steps{student_steps} ep{epoch}")
        for batch in pbar:
            batch=batch.to(device)
            noise=torch.randn_like(batch)*0.1
            input_latent=batch+noise 
            t_val = torch.full((batch.size(0),), float(student_steps)/float(cfg['teacher']['T']), device=device)
            pred = student(input_latent, t_val)
            loss=mse(pred,batch)

            opt.zero_grad();loss.backward();opt.step()
            global_step+=1
            pbar.set_postfix({'loss':loss.item()})
            if global_step % 1000 ==0:
                torch.save({'model':student.state_dict(), 'opt':opt.state_dict(), 'step':global_step}, ckpt_dir / f"student_steps{student_steps}_step{global_step}.pth")
    out=ckpt_dir/f"student_steps{student_steps}_final.pth"
    torch.save({'model':student.state_dict()},out)
    return str(out)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--teacher_ckpt', default=None)
    args = parser.parse_args()
    cfg = load_cfg(args.config)

    schedule=cfg['distillation']['halfing_schedule']
    final=cfg['distillation']['final_steps']
    prev_ckpt=None
    for s in schedule:
        if s<final: break
        prev_ckpt=train_student_stage(cfg,student_steps=s,init_ckpt=prev_ckpt,target_dir=cfg['data']['distill_target_dir'])
    print("Final student ckpt:",prev_ckpt)

if __name__=='__main__':
    main()