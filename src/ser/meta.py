import os, glob
import pandas as pd
from ser.common import DATA, EMOTIONS, FEATURES, ROOT

rows = []
for p in sorted(glob.glob(os.path.join(DATA, "Actor_*", "03-01-*.wav"))):
    f = os.path.basename(p)[:-4]
    _, _, e, i, s, r, a = f.split("-")
    rows.append(dict(file=f, path=os.path.relpath(p, DATA), actor=int(a), gender="M" if int(a) % 2 else "F",
                     emotion=EMOTIONS[int(e) - 1], label=int(e) - 1,
                     intensity="strong" if i == "02" else "normal", statement=int(s), repetition=int(r)))
meta = pd.DataFrame(rows)

h = pd.read_excel(ROOT / "data" / "human_ratings" / "s004.xlsx", sheet_name="Speech", header=1)
print(h["Modality"].value_counts().to_dict(), h["Filename"].iloc[-1])
h = h[h["Filename"].astype(str).str.startswith("03-01-")].copy()
h["file"] = h["Filename"].str.replace(".wav", "", regex=False)
h = h[["file", "PropCorr", "UHR"]].rename(columns={"PropCorr": "human_acc", "UHR": "human_uhr"})
meta = meta.merge(h, on="file", how="left")
FEATURES.mkdir(exist_ok=True)
meta.to_csv(FEATURES / "meta.csv", index=False)
print(meta.shape, "missing human:", int(meta["human_acc"].isna().sum()))
print(meta.groupby("emotion").size().to_dict())
print("human AO speech acc by intensity:", meta.groupby("intensity")["human_acc"].mean().round(3).to_dict(), "overall:", round(meta["human_acc"].mean(), 3))
