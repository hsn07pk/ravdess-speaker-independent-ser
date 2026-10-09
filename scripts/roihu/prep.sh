#!/bin/bash
set -e
mkdir -p /scratch/project_2020210/msohail/ravdess /scratch/project_2020210/msohail/hf
cd /scratch/project_2020210/msohail/ravdess
[ -f Audio_Speech_Actors_01-24.zip ] || wget -q -O Audio_Speech_Actors_01-24.zip "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
echo "bc696df654c87fed845eb13823edef8a  Audio_Speech_Actors_01-24.zip" | md5sum -c
[ -d Actor_24 ] || unzip -q Audio_Speech_Actors_01-24.zip
ls Actor_*/03-01-*.wav | wc -l
module load python-pytorch/2.10
export HF_HOME=/scratch/project_2020210/msohail/hf
cd ~/ser_ft
python -c "from transformers import AutoFeatureExtractor, WavLMForSequenceClassification; AutoFeatureExtractor.from_pretrained('microsoft/wavlm-large'); WavLMForSequenceClassification.from_pretrained('microsoft/wavlm-large', num_labels=8, use_safetensors=False); print('model ok')"
python finetune.py --fold 1 --data /scratch/project_2020210/msohail/ravdess --out ./results --prep-only
mkdir -p slurm_log results
echo PREPDONE
