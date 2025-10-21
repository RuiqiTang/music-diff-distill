import argparse,yaml,os 
from pathlib import Path
import numpy as np
import soundfile as sf
import torchaudio
from tqdm import tqdm
import torch 
from encodec import EncodecModel
from encodec.utils import convert_audio

def load_cfg(path):
    import yaml
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def ensure_mono(wav):
    if wav.ndim == 1:
        return wav
    if wav.shape[1] == 1:
        return wav[:,0]
    # mix to mono
    return np.mean(wav, axis=1)

def sliding_windows(wav, sr, window_sec, hop_sec):
    win_len = int(window_sec * sr)
    hop = int(hop_sec * sr)
    n = len(wav)
    if n <= win_len:
        yield 0, wav
    else:
        start = 0
        while start < n:
            end = min(start + win_len, n)
            chunk = wav[start:end]
            if len(chunk) < win_len:
                chunk = np.pad(chunk, (0, win_len - len(chunk)))
            yield start, chunk
            if end == n:
                break
            start += hop

def save_encodings(encodings,out_path,p,start,chunk):
    try:
        if isinstance(encodings, dict):
            # might contain 'codes' or 'quantized' keys
            tosave = {}
            for k,v in encodings.items():
                try:
                    tosave[k] = np.array(v)
                except Exception:
                    pass
            np.savez_compressed(out_path, **tosave, filename=str(p), start=start)
        elif isinstance(encodings, (list, tuple)):
            np.savez_compressed(out_path, *[np.array(x) for x in encodings], filename=str(p), start=start)
        else:
            np.savez_compressed(out_path, latent=np.array(encodings), filename=str(p), start=start)
    except Exception:
        # fallback: save raw wav
        np.savez_compressed(out_path, raw=np.array(chunk, dtype=np.float32), filename=str(p), start=start)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    wav_dir = Path(cfg['data']['wav_dir'])
    out_dir = Path(args.out or cfg['data']['latent_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    sr = cfg['data']['sample_rate']
    bitrate = cfg['encodec']['bitrate']
    window_sec = cfg['data']['window_sec']
    hop_sec = cfg['data']['hop_sec']

    enc=EncodecModel.encodec_model_24khz() if sr==24000 else EncodecModel.encodec_model_48khz()
    enc.set_target_bandwidth(bitrate/1000.)

    files=sorted([p for p in wav_dir.glob('**/*.wav')])
    idx=0
    for p in tqdm(files):
        wav,file_sr=sf.read(str(p))
        if file_sr!=sr:
            wav=torchaudio.functional.resample(torch.tensor(wav).float(),file_sr,sr).numpy()
        for start,chunk in sliding_windows(wav,sr,window_sec,hop_sec):
            audio=np.expand_dims(chunk.astype(np.float32),0)
            audio=convert_audio(audio,sr,sr,target_channels=1)

        # Use encodec API
        encodings=enc.encode(audio)
        out_path=out_dir/f"{p.stem}_{idx:06d}.npz"
        save_encodings(encodings,out_path,p,start,chunk)
        idx+=1

if __name__=="__main__":
    main()
