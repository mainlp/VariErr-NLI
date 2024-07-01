import json
import argparse
from typing import Dict
from collections import defaultdict
from pathlib import Path

import datasets

from predict_scorers import write_results

def get_reason_to_score(scoring_file):
    with open(scoring_file, "r") as f:
        scoring_data = json.load(f)
    
    reason_to_score = {k: float(v) for k, v in scoring_data.items()}


    return reason_to_score

def aggregate_scores(reason_to_score: Dict[str, float]) -> Dict[str, float]:
    label_to_score_sum = defaultdict(float)
    label_to_num_reasons = defaultdict(int)

    for reason_id, score in reason_to_score.items():
        label_id = reason_id[:reason_id.index("-")+2]
        label_to_score_sum[label_id] += score
        label_to_num_reasons[label_id] += 1
    
    # normalize 
    label_to_neg_avg_score = {}
    for label_id, score_sum in label_to_score_sum.items():
        label_to_neg_avg_score[label_id] = -(score_sum / label_to_num_reasons[label_id])


    return label_to_neg_avg_score



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scoring_file", help="The file containing the scoring data", type=Path)
    args = parser.parse_args()

    reason_to_score = get_reason_to_score(scoring_file=args.scoring_file)
    label_to_neg_avg_score = aggregate_scores(reason_to_score)

    dataset = datasets.load_dataset("mainlp/varierr")['train']

    prediction_dir = Path("predictions") / args.scoring_file.parent.name
    prediction_dir.mkdir(parents=True, exist_ok=True)

    with open((prediction_dir / "neg_avg_score").with_suffix(".json"), "w") as f:
        json.dump(label_to_neg_avg_score, f, indent=1)

    