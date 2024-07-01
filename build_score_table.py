import argparse
import json
from pathlib import Path
from collections import defaultdict

import datasets
from sklearn.metrics import average_precision_score, ndcg_score, precision_score, recall_score
import numpy as np
import pandas as pd

def recall_at_k(y_true, y_score, k):
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    idcs_pred = np.argsort(y_score)[::-1][:k]
    y_pred = np.zeros_like(y_true)
    y_pred[idcs_pred] = 1
    
    return recall_score(y_pred=y_pred, y_true=y_true)

def precision_at_k(y_true, y_score, k):
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    idcs_pred = np.argsort(y_score)[::-1][:k]
    
    y_true_k = y_true[idcs_pred]
    y_pred_k = np.ones_like(y_true_k)
    
    return precision_score(y_pred=y_pred_k, y_true=y_true_k)

def compute_metrics(id_to_score, id_to_ground_truth):
    ids = [id for id in id_to_ground_truth.keys() if id in id_to_score]
    y_true = np.array([id_to_ground_truth[id] for id in ids])
    y_score = np.array([float(id_to_score[id]) for id in ids])

    metrics = {}
    metrics["ap"] = average_precision_score(y_true=y_true, y_score=y_score)
    metrics["p@50"] = precision_at_k(y_true, y_score, 50)
    metrics["p@100"] = precision_at_k(y_true, y_score, 100)
    metrics["r@100"] = recall_at_k(y_true, y_score, 100)

    return metrics

def compute_random_baseline_recall_at_k(num_positive, num_total, k):
    """
    Expected num positive is the mean of a hypergeometric distribution (https://en.wikipedia.org/wiki/Hypergeometric_distribution)
    """

    expected_num_positive = num_positive * k / num_total
    recall = expected_num_positive / num_positive

    return recall

def compute_random_baseline_precision_at_k(num_positive, num_total, k):
    expected_num_positive = num_positive * k / num_total
    precision = expected_num_positive / k 

    return precision



def add_random_baseline(df_dict: defaultdict, id_to_ground_truth: dict) -> None:
    num_label_corrections = sum(id_to_ground_truth.values()) 
    df_dict["method"].append("random")
    df_dict["run"].append("random")
    df_dict["ap"].append(num_label_corrections / len(id_to_ground_truth))

    df_dict["r@100"].append(compute_random_baseline_recall_at_k(num_positive=num_label_corrections, num_total=len(id_to_ground_truth), k=100))
    df_dict["p@50"].append(compute_random_baseline_precision_at_k(num_positive=num_label_corrections, num_total=len(id_to_ground_truth), k=50))
    df_dict["p@100"].append(compute_random_baseline_precision_at_k(num_positive=num_label_corrections, num_total=len(id_to_ground_truth), k=100))


def prettify_results(df_mean_std: pd.DataFrame) -> str:
    rows = [f"Method\tAP\tP@50\tP@100\tR@100"]

    df_mean_std["mean"] = df_mean_std["mean"].apply(lambda x: round(x, 3)) * 100
    df_mean_std["std"] = df_mean_std["std"].apply(lambda x: round(x, 3)) * 100



    for method, row in df_mean_std.iterrows():
        row = [method]
        for metric in df_mean_std.columns.levels[1]:
            mean = df_mean_std["mean"][metric][method]
            std = df_mean_std["std"][metric][method]
            if "nan" not in str(std):
                row.append(f"${mean:.1f}\pm{std:.1f}$")
            else:
                row.append(f"${mean:.1f}$")
        rows.append("\t".join(row)) 

    return "\n".join(rows)


def build_score_table(dataset: datasets.Dataset, ground_truth="label_correction", rerank_single_label=False) -> str:
    id_to_ground_truth = {}

    def _get_label_correction_ground_truth():
        id_to_ground_truth = {}
        for instance in dataset:
            for label, count in instance['label_count_round_1'].items():
                if not count or (rerank_single_label and count > 1):
                    continue
                id_ = instance['id'] + "-" + label[0]
                id_to_ground_truth[id_] = label in instance['error_labels']

        return id_to_ground_truth


    def _get_ambiguous_ground_truth():
        id_to_ground_truth = {}
        for instance in dataset:
            for label, count in instance['label_count_round_1'].items():
                if not count or (rerank_single_label and count > 1):
                    continue
                id_ = instance['id'] + "-" + label[0]
                id_to_ground_truth[id_] = len(instance['label_set_round_2']) > 1 and label in instance['label_set_round_2']
        
        return id_to_ground_truth


    def _get_label_correction_or_ambiguous_ground_truth():
        id_to_ambiguous = _get_ambiguous_ground_truth()
        id_to_label_correction = _get_label_correction_ground_truth()

        return {id: id_to_ambiguous[id] or id_to_label_correction[id] for id in id_to_ambiguous}

    if ground_truth == "label_correction":
        id_to_ground_truth = _get_label_correction_ground_truth()
    elif ground_truth == "ambiguous":
        id_to_ground_truth = _get_ambiguous_ground_truth()
    elif ground_truth == "label_correction_or_ambiguous":
        id_to_ground_truth = _get_label_correction_or_ambiguous_ground_truth()
    else:
        raise ValueError(f"Unknown ground truth {ground_truth}")


    df_dict = defaultdict(list)
    prediction_dir = Path("predictions")
    for method_dir in prediction_dir.glob("*"):
        for run_file in method_dir.glob("*.json"):
            with open(run_file) as f:
                id_to_score = json.load(f)
            

            # filter out ids that are not in the label correction so that we can use subsets of the data
            id_to_score = {id: score for id, score in id_to_score.items() if id in id_to_ground_truth}
            metrics = compute_metrics(id_to_score, id_to_ground_truth)
            df_dict["method"].append(method_dir.name)
            df_dict["run"].append(run_file.stem)
            for metric, value in metrics.items():
                df_dict[metric].append(value)

    add_random_baseline(df_dict, id_to_ground_truth)
    df = pd.DataFrame(df_dict)


    df_mean = df.groupby(["method"]).mean(numeric_only=True)
    df_std = df.groupby(["method"]).std(numeric_only=True)
    df_mean_std = pd.concat([df_mean, df_std], axis=1, keys=["mean", "std"])
    pretty_results = prettify_results(df_mean_std)

    return pretty_results

# dataset = datasets.load_dataset("mainlp/varierr")["train"]
dataset = datasets.Dataset.from_json("dataset_maybe_new.json")

with open("results.tsv", "w") as f:
    f.write(build_score_table(dataset))

with open("results_ambiguous.tsv", "w") as f:
    f.write(build_score_table(dataset, ground_truth="ambiguous"))

with open("results_label_correction_or_ambiguous.tsv", "w") as f:
    f.write(build_score_table(dataset, ground_truth="label_correction_or_ambiguous"))

with open("results_single_label.tsv", "w") as f:
    f.write(build_score_table(dataset, rerank_single_label=True))
