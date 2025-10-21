import torch
import torch.nn as nn
from einops import rearrange

class SmallTimeEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Sequential(nn.Linear(dim, dim*2), nn.SiLU(), nn.Linear(dim*2, dim))

    def forward(self, t):
        half = t.shape[0]
        emb = t[:, None].repeat(1, 64)  # simplistic
        return self.lin(emb)

class SmallUNet1D(nn.Module):
    def __init__(self, in_ch=4, base_ch=64, time_emb_dim=128):
        super().__init__()
        self.in_conv = nn.Conv1d(in_ch, base_ch, kernel_size=3, padding=1)
        self.block1 = nn.Sequential(nn.Conv1d(base_ch, base_ch, 3, padding=1), nn.SiLU())
        self.down = nn.Conv1d(base_ch, base_ch*2, 4, stride=2, padding=1)
        self.mid = nn.Sequential(nn.Conv1d(base_ch*2, base_ch*2, 3, padding=1), nn.SiLU())
        self.up = nn.ConvTranspose1d(base_ch*2, base_ch, 4, stride=2, padding=1)
        self.out = nn.Conv1d(base_ch, in_ch, 1)
        self.time_mlp = SmallTimeEmbed(time_emb_dim)

    def forward(self, x, t):
        h = self.in_conv(x)
        h = self.block1(h)
        h = self.down(h)
        h = self.mid(h)
        h = self.up(h)
        return self.out(h)
