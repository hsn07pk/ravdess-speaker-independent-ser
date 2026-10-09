# Fine-tuning WavLM-L on CSC Roihu

These are the three files that ran on Roihu. `finetune.py` is self-contained: one call trains and tests one LOSO fold. To run them on another Slurm cluster, change the account, the partitions, the scratch paths and the module line.

1. Copy `finetune.py`, `prep.sh` and `run_finetune.sh` to `~/ser_ft` on the cluster.
2. `bash prep.sh` downloads RAVDESS (md5 checked), caches `microsoft/wavlm-large` and writes the trimmed 16 kHz waveforms once.
3. `sbatch run_finetune.sh` starts one array task per test actor. The reported run used `--array=1 --partition=gputest --time=00:15:00` for fold 1 and `--array=2-24%4 --time=00:15:00` for folds 2 to 24.
4. Copy `~/ser_ft/results` to `results/finetune/` and run `python -m ser.collect_ft results/finetune`.

Recipe (fixed before the run): WavLMForSequenceClassification with a learned weighted sum of all hidden layers, layerdrop off, CNN feature encoder frozen; AdamW with learning rate 3e-5 for the body and 1e-3 for the head and layer weights, weight decay 0.01, 10% linear warmup, gradient clipping 1.0; 10 epochs, batch 8, class-weighted cross-entropy, bf16 autocast, seed 0. In every fold the next two actor ids (one male, one female) are validation actors that pick the epoch; the other 21 actors are used for training.
