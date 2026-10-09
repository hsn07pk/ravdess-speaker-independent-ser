# Results

Everything in this folder is written by the code in `src/ser/`. Rows of every prediction array follow `features/meta.csv` (1440 clips); the class order is neutral, calm, happy, sad, angry, fearful, disgust, surprised. A run name is `<representation>_<variant>`, for example `wavlm_large_nested` (frozen WavLM-L, layer and C chosen per fold), `compare_lr`, `hubert_large_last`, `xlsr_300m_random5` or `wavlm_large_ft`.

## Tables

| File | Content |
|---|---|
| `summary.jsonl` | One line per run: outer and inner split, UAR, accuracy, macro-F1, the (layer, C) chosen in every outer fold, LR fits that hit the iteration limit |
| `table_all.csv` | Every run with UAR, accuracy, macro-F1, SD of the 24 per-actor UARs and the 95% actor-bootstrap interval |
| `wilcoxon_holm.csv` | The 23 paired comparisons: mean and median difference, wins, losses and ties over actors, Wilcoxon p and Holm-adjusted p |
| `ft_vs_frozen.csv` | Fine-tuned vs frozen WavLM-L, the planned comparison outside the Holm family |
| `classifiers.csv` | LR, SVM and RF on every feature set |
| `inflation.csv` | Nested LOSO against random 5-fold, per-speaker normalization and the best layer picked on the test results |
| `emobox_anchor.csv` | Last-layer probes on the EmoBox 6-fold split |
| `selection.csv` | How often each layer and C was chosen, and folds at the edges of the C grid |
| `sensitivity.csv` | Sensitivity runs: LR grid extended to C = 1000, SVM predict() against the probability argmax, and a rerun of the WavLM-L nested probe with the final code (`uar_v2` is the UAR of the reported run) |
| `human_vs_model.csv`, `human_bins.csv`, `human_by_emotion.csv`, `human_by_intensity.csv` | Comparison with the RAVDESS listeners on 7 options |
| `gender.csv` | Mean per-actor UAR of the female and the male actors |
| `sanity_permutation.json` | Labels permuted within each actor, 3 seeds, full nested selection |
| `e2v_plus_zeroshot.json` | emotion2vec+ large zero-shot on 7 options (contaminated, not a result) |

## Arrays

| Folder | Files | Shape |
|---|---|---|
| `predictions/` | `proba_<run>.npy`: out-of-fold class probabilities | (1440, 8) |
| | `labels_wavlm_large_perm_s<seed>.npy`: the permuted labels | (1440,) |
| | `e2v_plus_large_scores.npy`, `e2v_plus_large_labels.json`: scores of the emotion2vec+ head | (1440, 9) |
| `layers/` | `layer_<model>_<LL>.npy`: probabilities of the probe on layer LL, C chosen per fold | (1440, 8) |
| | `curve_<model>.npy`: LOSO UAR of every layer | (25,) or (33,) |
| `inner_cv/` | `inner_<run>.npy`: mean inner-CV UAR of every candidate | (folds, layers, C values) |
| `sensitivity/` | `sens_grid1e3_<run>.npy` (probabilities) and `sens_svm_predict_<run>.npy` (labels) of the sensitivity runs | (1440, 8) or (1440,) |
| `finetune/` | `fold_XX.json` (epoch log, chosen epoch, validation actors, test UAR, clip ids) and `proba_fold_XX.npy` (test-actor probabilities) of the 24 fine-tuning folds | (60, 8) |
