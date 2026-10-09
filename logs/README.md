# Logs

Raw output of the reported run.

- `environment/`: `conda list` of the two environments (main pipeline and emotion2vec) and the hardware and key package versions.
- `features/` (5.10.2026): handcrafted features (`handcrafted.log`), WavLM-L, HuBERT-L and XLS-R 300M embeddings (`ssl.log`, which ends with the error of a first Whisper attempt in fp16 that wrote nothing), the Whisper large-v3 encoder (`whisper.log`) and emotion2vec (`emotion2vec.log`).
- `evaluation/` (6.10.2026, fine-tuned row added 8.10.2026): one log per step of `scripts/evaluate.sh`, plus `analysis.log`, `sensitivity.log` and `figures.log`.
- `finetune/` (8.10.2026, CSC Roihu): stdout and stderr of every array task (`slurm/`), job accounting (`sacct.txt`) and data and model preparation (`prep.log`).
