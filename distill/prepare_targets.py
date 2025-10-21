import argparse,yaml
from pathlib import Path
import numpy as np
import encodec as EncodecModel
import soundfile as sf

def load_cfg(path):
    import yaml
    with open(path,'r') as f: return yaml.safe_load(f)

def encode_wave_to_latent(wave_np,sr,bitrate):
    enc = EncodecModel.encodec_model_24khz() if sr==24000 else EncodecModel.encodec_model_48khz()
    enc.set_target_bandwidth(bitrate/1000.0)
    encodings=enc.encode_24k(wave_np) if sr==24000 else enc.encode_48k(wave_np)
    return encodings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--teacher_gen_dir', default='./teacher_generated')
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    sr = cfg['data']['sample_rate']
    out_dir = Path(cfg['data']['distill_target_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    # iterate teacher_gen_dir for generated wavs or npz files
    gen_dir = Path(args.teacher_gen_dir)
    files = sorted(list(gen_dir.glob('*.wav')) + list(gen_dir.glob('*.npz')))
    idx = 0
    enc = EncodecModel.encodec_model_24khz() if sr==24000 else EncodecModel.encodec_model_48khz()
    enc.set_target_bandwidth(cfg['encodec']['bitrate']/1000.0)

    for p in files:
        if p.suffix=='.wav':
            wav,file_sr=sf.read(str(p))
            if file_sr!=sr:
                import librosa
                wav=librosa.resample(wav.astype('float32'),file_sr,sr)
            # Encodec expects shape (channels,N)
            audio=np.expand_dims(wav.astype('float32'),0)
            try:
                encodings = enc.encode_24k(audio) if sr==24000 else enc.encode_48k(audio)
            except Exception:
                encodings = enc.encode(audio)
            # Save encodings as npz
            np.savez_compressed(out_dir / f"target_{idx:06d}.npz", *[np.array(x) for x in encodings])
        else:
            data = np.load(str(p))
            # if data contains 'audio', encode
            if 'audio' in data.files:
                wav = data['audio']
                audio = np.expand_dims(wav.astype('float32'), 0)
                try:
                    encodings = enc.encode_24k(audio) if sr==24000 else enc.encode_48k(audio)
                except Exception:
                    encodings = enc.encode(audio)
                np.savez_compressed(out_dir / f"target_{idx:06d}.npz", *[np.array(x) for x in encodings])
            elif any(k.startswith('latent') for k in data.files) or any(k.startswith('code_') for k in data.files):
                # already latent-like, copy
                np.savez_compressed(out_dir / f"target_{idx:06d}.npz", **{k: data[k] for k in data.files})
        idx += 1

if __name__ == '__main__':
    main()



