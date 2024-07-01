#!/usr/bin/env bash

python sequence_classification_train.py --out_dir runs/42 --seed 42
python sequence_classification_train.py --out_dir runs/43 --seed 43
python sequence_classification_train.py --out_dir runs/44 --seed 44

python predict_scorers.py --outputs runs/42/train_outputs.pt
python predict_scorers.py --outputs runs/43/train_outputs.pt
python predict_scorers.py --outputs runs/44/train_outputs.pt

python predict_supervised.py --outputs runs/42/train_outputs.pt
python predict_supervised.py --outputs runs/43/train_outputs.pt
python predict_supervised.py --outputs runs/44/train_outputs.pt

# WARNING: This will use your OpenAI-API-key and will cost money.
# Uncomment at your own risk
# python predict_llms.py gpt-3.5-turbo
# python predict_llms.py gpt-4-1106-preview

python predict_baselines.py
