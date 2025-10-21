import argparse, yaml, os, math
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from teacher.load_teacher import load_teacher_pipline
from data_utils.dataset import LatentDataset
from tqdm import tqdm
import torch.optim as optim

def load_cfg(path):
    with open(path,'r') as f:
        import yaml
        return yaml.safe_load(f)

def get_internal_model(pipeline):
    # Attempts to extract the denoiser/unet from pipeline
    if hasattr(pipeline, 'unet'):
        return pipeline.unet
    # diffusers pipeline might have model called 'denoiser' or 'model'
    if hasattr(pipeline, 'model'):
        return pipeline.model
    raise RuntimeError("Cannot find unet internal model in pipeline. See README for fallback (use author repo loader).")

def train(cfg):
    device=torch.device(cfg['training']['device'])
    model_id=cfg['teacher']['hf_model_id']
    pipline=load_teacher_pipline(model_id,device=device)
    denoiser=get_internal_model(pipline)
    denoiser.train()

    dataset=LatentDataset(cfg['data']['latent_dir'])
    dataloader=DataLoader(
        dataset,
        batch_size=cfg['teacher']['batch_size'],
        shuffle=True,
        num_workers=cfg['training']['num_workers'],
        pin_memory=True
    )
    opt=optim.AdamW(denoiser.parameters(),lr=cfg['teacher']['lr'])
    scaler=torch.amp.GradScaler() if cfg['training'].get('mixed_precision',True) else None
    global_step=0

    ckpt_dir=Path(cfg['logging']['ckpt_dir'])
    ckpt_dir.mkdir(parents=True,exist_ok=True)
    T=cfg['teacher']['T']

    for epoch in range(cfg['teacher']['num_epochs']):
        pbar=tqdm(dataloader,desc=f'Teacher finetune epoch {epoch}')
        for batch in pbar:
            batch=batch.to(device)
            B=batch.shape[0]
            t=torch.randint(0,T,(B,),device=device)

            # forward through pipline's noise prediction path
            noise=torch.randn_like(batch)
            t_norm=t.float()/float(T)
            noisy=(1 - t_norm.view(-1,1,1)) * batch + t_norm.view(-1,1,1) * noise

            if scaler:
                with torch.amp.autocast():
                    pred=denoiser(noisy,t_norm)
                    loss=torch.nn.functional.mse_loss(pred,noise)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
            else:
                pred=denoiser(noisy,t_norm)
                loss=torch.nn.functional.mse_loss(pred,noise)
                loss.backward();opt.step();opt.zero_grad()
            pbar.set_postfix({'loss':loss.item()})
            if global_step % cfg['logging'].get('sample_every_steps', 1000) == 0:
                torch.save({'model':denoiser.state_dict(), 'opt':opt.state_dict(), 'step':global_step}, ckpt_dir / f"teacher_ft_step{global_step}.pth")
        torch.save({'model':denoiser.state_dict(), 'opt':opt.state_dict(), 'epoch':epoch}, ckpt_dir / f"teacher_ft_epoch{epoch}.pth")
        print('Teacher fine-tune finished')
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    train(cfg)