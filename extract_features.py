import os
import numpy as np, pandas as pd, librosa, parselmouth
from joblib import Parallel, delayed

N_PRAAT = 7 
N_MFCC = 26

def praat_features(snd):
    """Pitch (3) + formants F1/F2 averaged over voiced frames (4)."""
    pitch = snd.to_pitch()
    f0 = pitch.selected_array["frequency"]
    times = pitch.xs()
    voiced = f0 > 0                      # 0 = no pitch detected
    f0v = f0[voiced]
    if len(f0v) == 0:
        return np.full(N_PRAAT, np.nan)
    formant = snd.to_formant_burg()
    f1 = np.array([formant.get_value_at_time(1, t) for t in times[voiced]])
    f2 = np.array([formant.get_value_at_time(2, t) for t in times[voiced]])
    return np.array([f0v.mean(), f0v.std(), f0v.max() - f0v.min(),
                     np.nanmean(f1), np.nanstd(f1), np.nanmean(f2), np.nanstd(f2)])

def extract_features(path):
    try:
        y, sr = librosa.load(path, sr=16000)
        snd = parselmouth.Sound(y.astype("float64"), sampling_frequency=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_feats = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
        return praat_features(snd), mfcc_feats
    except Exception as e:
        print("failed:", path, e)
        return np.full(N_PRAAT, np.nan), np.full(N_MFCC, np.nan)

def run(csv, audio_dir, prefix, limit=None):
    df = pd.read_csv(csv)
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(extract_features)(os.path.join(audio_dir, p)) for p in df.path[:limit])
    np.save(prefix + "_praat.npy", np.vstack([r[0] for r in results]))
    np.save(prefix + "_mfcc.npy", np.vstack([r[1] for r in results]))

if __name__ == "__main__":
    run("data/train_filtered.csv", "data/train", "data/train")