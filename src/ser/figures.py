import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from ser.common import FIGURES, RESULTS, load_meta, result
from ser.probe import scores

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42,
                     "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7})
meta = load_meta()
y = meta["label"].values
res, figd = RESULTS, FIGURES
os.makedirs(figd, exist_ok=True)
P = lambda n: np.load(result(f"proba_{n}.npy"))
uar = lambda n: scores(y, P(n))["uar"]
models = [("xlsr_300m", "XLS-R 300M"), ("wavlm_large", "WavLM Large"), ("whisper_large_v3", "Whisper-large-v3 enc."), ("hubert_large", "HuBERT Large")]
style = {"xlsr_300m": ("#0072B2", "o", "-"), "wavlm_large": ("#E69F00", "s", "--"),
         "whisper_large_v3": ("#009E73", "^", "-."), "hubert_large": ("#D55E00", "D", ":")}

fig, ax = plt.subplots(figsize=(3.5, 2.5))
for name, label in models:
    c = np.load(result(f"curve_{name}.npy")) * 100
    x = np.arange(len(c)) / (len(c) - 1)
    col, mk, ls = style[name]
    ax.plot(x, c, color=col, marker=mk, ms=2.5, lw=1.1, ls=ls, label=label)
for n, label, ls in [("compare_lr", "ComParE + LR", (0, (4, 2))), ("egemaps_lr", "eGeMAPS + LR", (0, (1, 1.5)))]:
    ax.axhline(uar(n) * 100, color="gray", ls=ls, lw=0.9, label=label)
ax.set_xlabel("relative layer depth (0: input to layer 1, 1: last layer)")
ax.set_ylabel("LOSO UAR (%)")
ax.set_ylim(40, 90)
ax.set_xlim(-0.02, 1.02)
ax.grid(alpha=0.3, lw=0.4)
ax.legend(ncol=2, loc="lower right", frameon=False, handlelength=2.6, columnspacing=1.0)
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(figd, "layer_curves.pdf"), metadata={"CreationDate": None})
fig.savefig(os.path.join(figd, "layer_curves.png"), dpi=300)

best = max([f"{m}_nested" for m, _ in models], key=uar)
names = dict(models)
short = ["neu", "cal", "hap", "sad", "ang", "fea", "dis", "sur"]
fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.7), gridspec_kw={"width_ratios": [1, 1, 1.1]})
for ax, (n, title) in zip(axes[:2], [("compare_lr", "(a) ComParE + LR"), (best, "(b) " + names[best.replace("_nested", "")] + " + LR")]):
    cm = confusion_matrix(y, P(n).argmax(1), normalize="true") * 100
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=100)
    for i in range(8):
        for j in range(8):
            ax.text(j, i, f"{cm[i, j]:.0f}", ha="center", va="center", fontsize=6.5, color="white" if cm[i, j] > 55 else "black")
    ax.set_xticks(range(8), short)
    ax.set_yticks(range(8), short)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"{title}, UAR {uar(n) * 100:.1f}%")
bins = pd.read_csv(os.path.join(res, "human_bins.csv"))
order = ["0-30%", "40-60%", "70-90%", "100%"]
ticks = ["0 to 30%", "40 to 60%", "70 to 90%", "100%"]
ax = axes[2]
for n, col, mk, ls, label in [(best, "#0072B2", "o", "-", names[best.replace("_nested", "")]),
                               ("whisper_large_v3_nested", "#009E73", "^", "-.", names["whisper_large_v3"]),
                               ("compare_lr", "gray", "s", ":", "ComParE + LR")]:
    b = bins[bins["system"] == n].set_index("human_bin").loc[order]
    ax.plot(range(4), b["model"].values * 100, color=col, marker=mk, ms=3, lw=1.1, ls=ls, label=label)
b = bins[bins["system"] == best].set_index("human_bin").loc[order]
ax.plot(range(4), b["human"].values * 100, color="black", ls="--", lw=0.9, marker="x", ms=3.5, label="human listeners")
ax.set_xticks(range(4), [f"{t}\n(n = {c})" for t, c in zip(ticks, b["n"].values)])
ax.set_xlabel("share of the 10 listeners who were correct")
ax.set_ylabel("7-option accuracy (%)")
ax.set_ylim(0, 100)
ax.grid(alpha=0.3, lw=0.4)
ax.legend(frameon=False, loc="lower right")
ax.set_title("(c) accuracy by human difficulty")
fig.tight_layout(pad=0.3, w_pad=1.0)
fig.savefig(os.path.join(figd, "confusions_human.pdf"), metadata={"CreationDate": None})
fig.savefig(os.path.join(figd, "confusions_human.png"), dpi=300)
print("figures written; best =", best)
