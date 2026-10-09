import os, sys, time, json
import numpy as np
import librosa
from funasr import AutoModel
from ser.common import FEATURES, load_meta

name, model_id = sys.argv[1], sys.argv[2]
meta = load_meta()
if os.environ.get("LIMIT"):
    meta = meta.head(int(os.environ["LIMIT"]))
model = AutoModel(model=model_id, hub="hf", disable_update=True)

feats, scores, labels = [], [], None
t0 = time.time()
for i, p in enumerate(meta["path"]):
    y, _ = librosa.load(p, sr=16000, mono=True)
    yt, _ = librosa.effects.trim(y, top_db=30)
    y = (yt if yt.size >= 1600 else y).astype(np.float32)
    r = model.generate(y, granularity="utterance", extract_embedding=True, disable_pbar=True)[0]
    feats.append(np.asarray(r["feats"], dtype=np.float32))
    if "scores" in r:
        scores.append(np.asarray(r["scores"], dtype=np.float32))
        labels = r["labels"]
    if (i + 1) % 200 == 0:
        print(f"{name} {i + 1}/{len(meta)} {time.time() - t0:.0f}s", flush=True)
np.save(FEATURES / f"{name}.npy", np.stack(feats))
if scores:
    np.save(FEATURES / f"{name}_scores.npy", np.stack(scores))
    with open(FEATURES / f"{name}_labels.json", "w") as f:
        json.dump(labels, f, ensure_ascii=False)
print(name, np.stack(feats).shape, f"{(time.time() - t0) / len(meta):.3f} s/clip", labels, flush=True)
