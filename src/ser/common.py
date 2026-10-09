import os
from pathlib import Path
import pandas as pd

ROOT = Path(os.environ.get("SER_ROOT", Path(__file__).resolve().parents[2]))
DATA = Path(os.environ.get("RAVDESS_DIR", ROOT / "data" / "RAVDESS_speech"))
FEATURES = ROOT / "features"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
LOGS = ROOT / "logs"
DOCS = ROOT / "docs"
EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
SUBDIRS = {"proba": "predictions", "labels": "predictions", "layer": "layers", "curve": "layers", "inner": "inner_cv", "sens": "sensitivity"}


def result(name):
    sub = SUBDIRS.get(name.split("_")[0]) if name.endswith(".npy") else None
    path = RESULTS / sub / name if sub else RESULTS / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_meta():
    meta = pd.read_csv(FEATURES / "meta.csv")
    meta["path"] = [str(DATA / p) for p in meta["path"]]
    return meta


def load_audio(path, sr=16000):
    import librosa
    y, _ = librosa.load(path, sr=sr, mono=True)
    yt, _ = librosa.effects.trim(y, top_db=30)
    return yt if yt.size >= sr // 10 else y
