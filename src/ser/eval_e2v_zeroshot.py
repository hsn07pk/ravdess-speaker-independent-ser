import os, json
import numpy as np
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from ser.common import FEATURES, RESULTS, load_meta

meta = load_meta()
y7 = np.where(meta["label"].values == 1, 0, meta["label"].values)
feat = FEATURES
S = np.load(os.path.join(feat, "e2v_plus_large_scores.npy"))
labels = [l.split("/")[-1] for l in json.load(open(os.path.join(feat, "e2v_plus_large_labels.json")))]
to_ravdess = {"neutral": 0, "happy": 2, "sad": 3, "angry": 4, "fearful": 5, "disgusted": 6, "surprised": 7}
lut = np.array([to_ravdess.get(l, -1) for l in labels])
cols = np.where(lut >= 0)[0]
pred_emotions = lut[cols[S[:, cols].argmax(1)]]
pred_all = lut[S.argmax(1)]
res = dict(acc7=accuracy_score(y7, pred_emotions), uar7=balanced_accuracy_score(y7, pred_emotions),
           acc7_all9=accuracy_score(y7, pred_all), other_or_unk=float((pred_all < 0).mean()))
print(json.dumps({k: round(float(v), 4) for k, v in res.items()}))
json.dump(res, open(RESULTS / "e2v_plus_zeroshot.json", "w"))
