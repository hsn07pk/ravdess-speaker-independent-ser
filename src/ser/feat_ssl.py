import os, sys, time
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModel, WhisperFeatureExtractor, WhisperModel
from ser.common import FEATURES, load_meta, load_audio

name, model_id = sys.argv[1], sys.argv[2]
dev = "mps" if torch.backends.mps.is_available() else "cpu"
meta = load_meta()
if os.environ.get("LIMIT"):
    meta = meta.head(int(os.environ["LIMIT"]))
whisper = "whisper" in model_id
if whisper:
    fe = WhisperFeatureExtractor.from_pretrained(model_id)
    model = WhisperModel.from_pretrained(model_id).get_encoder()
else:
    fe = AutoFeatureExtractor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, use_safetensors=False)
model = model.float().to(dev).eval()

feats = None
t0 = time.time()
with torch.inference_mode():
    for i, p in enumerate(meta["path"]):
        y = load_audio(p)
        if whisper:
            x = fe(y, sampling_rate=16000, return_tensors="pt").input_features.to(dev)
            n = int(np.ceil(len(y) / 320))
        else:
            x = fe(y, sampling_rate=16000, return_tensors="pt").input_values.to(dev)
            n = None
        hs = model(x, output_hidden_states=True).hidden_states
        h = torch.stack([s[0, :n] for s in hs]).float()
        v = torch.cat([h.mean(1), h.std(1)], -1).cpu().numpy()
        if feats is None:
            feats = np.zeros((len(meta),) + v.shape, dtype=np.float32)
        feats[i] = v
        if (i + 1) % 100 == 0:
            print(f"{name} {i + 1}/{len(meta)} {time.time() - t0:.0f}s", flush=True)
sec = (time.time() - t0) / len(meta)
np.save(FEATURES / f"{name}.npy", feats)
with open(FEATURES / f"{name}_timing.txt", "w") as f:
    f.write(f"{model_id}\t{dev}\t{sec:.4f} s/clip\t{sum(p.numel() for p in model.parameters())} params\n")
print(name, feats.shape, f"{sec:.3f} s/clip", flush=True)
