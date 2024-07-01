from pathlib import Path
import argparse
import json

import datasets
import torch
import numpy as np

ID_TO_LABEL = {
    0: "contradiction",
    1: "entailment",
    2: "neutral",
}

LABEL_TO_ID = {v: k for k, v in ID_TO_LABEL.items()}



def get_probs_per_epoch(outputs: torch.Tensor) -> list[np.ndarray]:
    probs_per_epoch = []
    for output in outputs:
        probs_per_epoch.append(torch.sigmoid(torch.tensor(output.predictions)).numpy())

    return probs_per_epoch

def get_logits_per_epoch(outputs: torch.Tensor) -> list[np.ndarray]:
    logits_per_epoch = []
    for output in outputs:
        logits_per_epoch.append(output.predictions)

    return logits_per_epoch

def predict_datamap_means(outputs: torch.Tensor) -> np.ndarray:
    probs_per_epoch = get_probs_per_epoch(outputs)
    means = np.mean(probs_per_epoch, axis=0)

    return means

def predict_datamap_stds(outputs: torch.Tensor) -> np.ndarray:
    probs_per_epoch = get_probs_per_epoch(outputs)
    stds = np.std(probs_per_epoch, axis=0)

    return stds

def write_results(predictions: np.ndarray, dataset: datasets.Dataset, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    for prediction, instance in zip(predictions, dataset):
        for idx_label, label in ID_TO_LABEL.items():
            if label in instance['label_set_round_1']:
                results[f"{instance['id']}-{label[0]}"] = float(prediction[idx_label])
    
    with open(output_path.with_suffix(".json"), 'w') as f:
        json.dump(results, f, indent=1)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs', type=Path, required=True)
    args = parser.parse_args()

    output_dir = Path("predictions")


    dataset = datasets.load_dataset("mainlp/varierr")['train']
    outputs = torch.load(str(args.outputs))
    model_name = args.outputs.parent.name

    datamap_means = predict_datamap_means(outputs)
    write_results(predictions=-datamap_means, dataset=dataset, output_path=output_dir / 'dm_mean' / model_name)

    datamap_stds = predict_datamap_stds(outputs)
    write_results(predictions=datamap_stds, dataset=dataset, output_path=output_dir / 'dm_std' / model_name)

    


    