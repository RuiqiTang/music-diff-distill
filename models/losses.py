import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchaudio

class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self, fft_sizes=None,hop_size=None,win_lengths=None):
        super().__init__()
        self.fft_sizes=fft_sizes or [1024,2048,512]
        self.hop_sizes=hop_size or [120,240,50]
        self.win_lengths=win_lengths or [600,1200,240]
        self.window=lambda n: torch.hann_window(n)
    
    def forward(self,x:Tensor,y:Tensor):
        # x,y:(B,T) or (B,1,T)
        if x.dim()==3:
            x=x.unsqueeze(1)
        if y.dim()==3:
            y=y.unsqueeze(1)

        loss=0.
        for n_fft,hop,win in zip(self.fft_sizes,self.hop_sizes,self.win_lengths):
            win_tensor=self.window(win).to(x.device)
            X=torch.stft(x,n_fft,hop_length=hop,win_length=win,window=win_tensor,return_complex=True)
            Y=torch.stft(y,n_fft,hop_length=hop,win_length=win,window=win_tensor,return_complex=True)
            X_mag=torch.abs(X)
            Y_mag=torch.abs(Y)
            sc_loss=torch.mean((X_mag-Y_mag)**2)
            mag_loss=torch.mean(torch.abs(X_mag-Y_mag))
            loss+=sc_loss+mag_loss
        return loss

# perceptual embedding placeholder
class PerceptualLoss(nn.Module):
    def __init__(self, emb_model=None):
        super().__init__()
        self.emb_model=emb_model
    
    def forward(self,x:Tensor,y:Tensor):
        if self.emb_model is None:
            return torch.tensor(0.,device=x.device)
        ex=self.emb_model(x)
        ey=self.emb_model(y)
        return F.mse_loss(ex,ey)