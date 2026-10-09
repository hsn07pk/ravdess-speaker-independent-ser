import os
import numpy as np
import pandas as pd
import librosa
import opensmile
from joblib import Parallel, delayed
from ser.common import FEATURES, load_meta, load_audio

SR = 22050


def librosa_feats(path):
    y = load_audio(path, sr=SR)
    f = {}
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=40)
    for i in range(40):
        f[f"mfcc_mean_{i}"] = mfcc[i].mean()
        f[f"mfcc_std_{i}"] = mfcc[i].std()
    f0, voiced, _ = librosa.pyin(y, fmin=65, fmax=500, sr=SR)
    f0 = f0[voiced & np.isfinite(f0)]
    f["f0_mean"] = f0.mean() if f0.size else 0.0
    f["f0_std"] = f0.std() if f0.size else 0.0
    f["f0_min"] = f0.min() if f0.size else 0.0
    f["f0_max"] = f0.max() if f0.size else 0.0
    f["f0_range"] = f["f0_max"] - f["f0_min"]
    f["voiced_frac"] = voiced.mean()
    rms = librosa.feature.rms(y=y)[0]
    f["rms_mean"], f["rms_std"] = rms.mean(), rms.std()
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    f["zcr_mean"], f["zcr_std"] = zcr.mean(), zcr.std()
    chroma = librosa.feature.chroma_stft(y=y, sr=SR).mean(axis=1)
    for i in range(12):
        f[f"chroma_{i}"] = chroma[i]
    contrast = librosa.feature.spectral_contrast(y=y, sr=SR).mean(axis=1)
    for i in range(len(contrast)):
        f[f"contrast_{i}"] = contrast[i]
    cent = librosa.feature.spectral_centroid(y=y, sr=SR)[0]
    f["cent_mean"], f["cent_std"] = cent.mean(), cent.std()
    f["bandwidth_mean"] = librosa.feature.spectral_bandwidth(y=y, sr=SR)[0].mean()
    f["rolloff_mean"] = librosa.feature.spectral_rolloff(y=y, sr=SR)[0].mean()
    return f


def smile_feats(path, feature_set):
    smile = opensmile.Smile(feature_set=feature_set, feature_level=opensmile.FeatureLevel.Functionals)
    y = load_audio(path, sr=16000)
    return smile.process_signal(y, 16000).iloc[0].to_dict()


meta = load_meta()
if os.environ.get("LIMIT"):
    meta = meta.head(int(os.environ["LIMIT"]))
paths = list(meta["path"])
out = FEATURES

df = pd.DataFrame(Parallel(n_jobs=12)(delayed(librosa_feats)(p) for p in paths))
df.insert(0, "file", meta["file"].values)
df.to_csv(os.path.join(out, "handcrafted.csv"), index=False)
print("handcrafted", df.shape, flush=True)

for name, fs in [("egemaps", opensmile.FeatureSet.eGeMAPSv02), ("compare", opensmile.FeatureSet.ComParE_2016)]:
    df = pd.DataFrame(Parallel(n_jobs=12)(delayed(smile_feats)(p, fs) for p in paths))
    df.insert(0, "file", meta["file"].values)
    df.to_csv(os.path.join(out, f"{name}.csv"), index=False)
    print(name, df.shape, flush=True)
