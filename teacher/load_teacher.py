from huggingface_hub import hf_hub_download
from diffusers import AudioLDMPipeline
import torch

def load_teacher_pipline(model_id='cvssp/audioldm-m-full',device='cuda'):
    pipline=AudioLDMPipeline.from_pretrained(model_id)
    pipline.to(device)
    return pipline