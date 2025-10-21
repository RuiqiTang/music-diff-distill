import numpy as np
import librosa
import soundfile as sf

def load_wav(path, sr=24000):
    wav, file_sr = sf.read(path)
    if wav.ndim>1:
        wav = np.mean(wav, axis=1)
    if file_sr != sr:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=file_sr, target_sr=sr)
    return wav.astype(np.float32)

def write_wav(path, wav, sr):
    sf.write(path, wav, sr)
