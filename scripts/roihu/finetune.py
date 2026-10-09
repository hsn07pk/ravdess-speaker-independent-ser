import os, sys, json, time, glob, argparse, pickle
import numpy as np
import torch
import librosa
from transformers import AutoFeatureExtractor, WavLMForSequenceClassification, get_linear_schedule_with_warmup

p = argparse.ArgumentParser()
p.add_argument("--fold", type=int, required=True)
p.add_argument("--data", required=True)
p.add_argument("--out", required=True)
p.add_argument("--model", default="microsoft/wavlm-large")
p.add_argument("--epochs", type=int, default=10)
p.add_argument("--bs", type=int, default=8)
p.add_argument("--lr", type=float, default=3e-5)
p.add_argument("--head-lr", type=float, default=1e-3)
p.add_argument("--limit-actors", type=int, default=0)
p.add_argument("--prep-only", action="store_true")
args = p.parse_args()

torch.manual_seed(0)
np.random.seed(0)
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def load_data(root):
    cache = os.path.join(root, "ravdess_16k_trim.pkl")
    if os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    items = []
    for path in sorted(glob.glob(os.path.join(root, "Actor_*", "03-01-*.wav"))):
        f = os.path.basename(path)[:-4]
        parts = f.split("-")
        y, _ = librosa.load(path, sr=16000, mono=True)
        yt, _ = librosa.effects.trim(y, top_db=30)
        y = yt if yt.size >= 1600 else y
        items.append((f, int(parts[6]), int(parts[2]) - 1, y.astype(np.float32)))
    pickle.dump(items, open(cache, "wb"))
    return items


items = load_data(args.data)
if args.prep_only:
    print("cached", len(items), "clips", sum(len(it[3]) for it in items) / 16000 / 3600, "hours")
    sys.exit(0)
test_actor = args.fold
val_actors = [test_actor % 24 + 1, (test_actor + 1) % 24 + 1]
if args.limit_actors:
    keep = {test_actor, *val_actors, *[a for a in range(1, 25) if a not in (test_actor, *val_actors)][:args.limit_actors]}
    items = [it for it in items if it[1] in keep]
train = [it for it in items if it[1] != test_actor and it[1] not in val_actors]
val = [it for it in items if it[1] in val_actors]
test = [it for it in items if it[1] == test_actor]

fe = AutoFeatureExtractor.from_pretrained(args.model)
model = WavLMForSequenceClassification.from_pretrained(args.model, num_labels=8, use_weighted_layer_sum=True, layerdrop=0.0, use_safetensors=False)
model.freeze_feature_encoder()
model.to(dev)

counts = np.bincount([it[2] for it in train], minlength=8)
weights = torch.tensor(counts.sum() / (8 * counts), dtype=torch.float32, device=dev)
loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
head = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("wavlm.")]
body = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("wavlm.")]
opt = torch.optim.AdamW([{"params": body, "lr": args.lr}, {"params": head, "lr": args.head_lr}], weight_decay=0.01)
steps = args.epochs * int(np.ceil(len(train) / args.bs))
sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
amp = dict(device_type="cuda", dtype=torch.bfloat16) if dev == "cuda" else None


def batches(data, shuffle):
    idx = np.random.permutation(len(data)) if shuffle else np.arange(len(data))
    for i in range(0, len(idx), args.bs):
        b = [data[j] for j in idx[i:i + args.bs]]
        x = fe([it[3] for it in b], sampling_rate=16000, padding=True, return_attention_mask=True, return_tensors="pt")
        yield x["input_values"].to(dev), x["attention_mask"].to(dev), torch.tensor([it[2] for it in b], device=dev)


def forward(x, m):
    if amp:
        with torch.autocast(**amp):
            return model(input_values=x, attention_mask=m).logits.float()
    return model(input_values=x, attention_mask=m).logits


def predict(data):
    model.eval()
    out = []
    with torch.no_grad():
        for x, m, _ in batches(data, False):
            out.append(torch.softmax(forward(x, m), -1).cpu().numpy())
    model.train()
    return np.concatenate(out)


def uar(y, pred):
    return float(np.mean([np.mean(pred[y == k] == k) for k in np.unique(y)]))


log, best, best_state = [], -1.0, None
t0 = time.time()
model.train()
for ep in range(args.epochs):
    tot = 0.0
    for x, m, y in batches(train, True):
        loss = loss_fn(forward(x, m), y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        tot += loss.item() * len(y)
    pv = predict(val)
    v = uar(np.array([it[2] for it in val]), pv.argmax(1))
    log.append(dict(epoch=ep + 1, train_loss=tot / len(train), val_uar=v, sec=time.time() - t0))
    print(json.dumps(log[-1]), flush=True)
    if v > best:
        best = v
        best_state = {k: t.detach().to("cpu", copy=True) for k, t in model.state_dict().items()}

model.load_state_dict(best_state)
pt = predict(test)
yt = np.array([it[2] for it in test])
os.makedirs(args.out, exist_ok=True)
np.save(os.path.join(args.out, f"proba_fold_{test_actor:02d}.npy"), pt)
res = dict(fold=test_actor, val_actors=val_actors, n_train=len(train), n_val=len(val), n_test=len(test),
           best_val_uar=best, best_epoch=int(np.argmax([l["val_uar"] for l in log]) + 1),
           test_uar=uar(yt, pt.argmax(1)), test_acc=float(np.mean(pt.argmax(1) == yt)),
           files=[it[0] for it in test], log=log, device=dev, minutes=(time.time() - t0) / 60)
json.dump(res, open(os.path.join(args.out, f"fold_{test_actor:02d}.json"), "w"), indent=1)
print(json.dumps({k: res[k] for k in ["fold", "best_epoch", "best_val_uar", "test_uar", "test_acc", "minutes"]}), flush=True)
