import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Utilities
def exists(x):
    return x is not None

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        half = self.dim // 2
        emb = torch.logspace(-4, 4, half, device=x.device)
        args = x[..., None] * emb
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim=None):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_ch)) if time_emb_dim else None
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t=None):
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        if self.time_mlp is not None and t is not None:
            h = h + self.time_mlp(t).unsqueeze(-1)
        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)
        return h + self.skip(x)

class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv1d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)

class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.ConvTranspose1d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)

class LatentUNet(nn.Module):
    def __init__(self, in_ch=4, base_ch=256, ch_mults=(1,2,4), time_emb_dim=512, attn_resolutions=(16,)):
        super().__init__()
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(time_emb_dim//4), nn.Linear(time_emb_dim//4, time_emb_dim), nn.SiLU(), nn.Linear(time_emb_dim, time_emb_dim))
        self.input_conv = nn.Conv1d(in_ch, base_ch, 3, padding=1)
        # down
        self.downs = nn.ModuleList()
        ch = base_ch
        for mult in ch_mults:
            out_ch = base_ch * mult
            self.downs.append(nn.ModuleList([
                ResidualBlock(ch, out_ch, time_emb_dim=time_emb_dim),
                ResidualBlock(out_ch, out_ch, time_emb_dim=time_emb_dim),
                Downsample(out_ch)
            ]))
            ch = out_ch
        # middle
        self.mid = nn.ModuleList([
            ResidualBlock(ch, ch, time_emb_dim=time_emb_dim),
            ResidualBlock(ch, ch, time_emb_dim=time_emb_dim),
        ])
        # up
        self.ups = nn.ModuleList()
        for mult in reversed(ch_mults):
            out_ch = base_ch * mult
            self.ups.append(nn.ModuleList([
                ResidualBlock(ch + out_ch, out_ch, time_emb_dim=time_emb_dim),
                ResidualBlock(out_ch, out_ch, time_emb_dim=time_emb_dim),
                Upsample(out_ch)
            ]))
            ch = out_ch
        self.out = nn.Sequential(nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv1d(ch, in_ch, 3, padding=1))

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        hs = []
        h = self.input_conv(x)
        for block1, block2, down in self.downs:
            h = block1(h, t_emb)
            h = block2(h, t_emb)
            hs.append(h)
            h = down(h)
        for m in self.mid:
            h = m(h, t_emb)
        for (block1, block2, up), skip in zip(self.ups, reversed(hs)):
            # upsample
            h = up(h)
            # concat skip
            h = torch.cat([h, skip], dim=1)
            h = block1(h, t_emb)
            h = block2(h, t_emb)
        return self.out(h)
