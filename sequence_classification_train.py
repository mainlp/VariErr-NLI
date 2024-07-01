import json
import math
import os
import pickle
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Union, Optional, List, Dict
from collections import Counter, defaultdict

import datasets
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import precision_score, recall_score, accuracy_score, average_precision_score, f1_score
from sklearn.model_selection import KFold, train_test_split
import transformers
import argparse
import torch
from torch import nn
from tqdm import tqdm, trange
from transformers import PreTrainedTokenizerBase, Trainer
import numpy as np
from transformers import DataCollatorWithPadding
from transformers.utils import PaddingStrategy

base_dir = Path(__file__).parent

# set wandb name
os.environ["WANDB_PROJECT"] = "tree"


class OutputsGetterCallback(transformers.TrainerCallback):
    def __init__(self, train_data):
        self.train_data = train_data
        self.train_outputs = []
        self.trainer = None

    def on_epoch_end(self, args, state, control, **kwargs):
        self.train_outputs.append(self.trainer.predict(self.train_data))


def compute_metrics(p):
    predictions, labels = p
    predictions = predictions > 0

    precision = precision_score(labels, predictions, average="micro")
    recall = recall_score(labels, predictions, average="micro")
    f1 = f1_score(labels, predictions, average="micro")
    accuracy = accuracy_score(labels, predictions)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--n_epochs", default=10, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--lr", default=3e-5, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--model_name", default="distilroberta-base")
    args = parser.parse_args()

    transformers.set_seed(args.seed)

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name, add_prefix_space=True)
    dataset = datasets.load_dataset("mainlp/varierr")['train']

    labels = set()
    for counter in dataset['label_count_round_1']:
        labels.update(counter.keys())

    label2id = {l: i for i, l in enumerate(sorted(labels))}
    id2label = {v: k for k, v in label2id.items()}


    def add_labels(instance):
        labels = np.zeros(len(label2id))
        for label, count in instance["label_count_round_1"].items():
            if count and count > 0:
                labels[label2id[label]] = 1
        instance["labels"] = labels

        return instance


    def tokenize(instances):
        text_pairs = list(zip(instances["context"], instances["statement"]))
        features = tokenizer(text_pairs)

        return features


    dataset = dataset.map(add_labels) \
        .map(tokenize, batched=True)

    model = transformers.AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=len(label2id),
                                                                            label2id=label2id, id2label=id2label)

    outputs_getter = OutputsGetterCallback(train_data=dataset)

    if torch.cuda.is_available():
        batch_size_denominator = max(1,
                                     torch.cuda.device_count())  # Make batch size consistent across computation environments with different numbers of GPUs
    else:
        batch_size_denominator = 1
    batch_size = args.batch_size
    num_epochs = args.n_epochs

    training_args = transformers.TrainingArguments(
        output_dir=args.out_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=batch_size // batch_size_denominator,
        num_train_epochs=num_epochs,
        fp16=True if torch.cuda.is_available() else False,
        report_to=["wandb"],
        evaluation_strategy="no",
        save_strategy="no",
        load_best_model_at_end=True,
        use_mps_device=False,
        save_total_limit=1,
        # eval_steps=10,

    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        callbacks=[outputs_getter]
    )
    outputs_getter.trainer = trainer

    trainer.train()

    # save model + outputs
    model.save_pretrained(args.out_dir)
    torch.save(outputs_getter.train_outputs, args.out_dir / "train_outputs.pt")
