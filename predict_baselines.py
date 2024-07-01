from pathlib import Path
import json

import datasets

prediction_dir = Path("predictions")

dataset = datasets.load_dataset("mainlp/varierr")['train']
id_to_negative_label_count = {}
for instance in dataset:
    for label, count in instance['label_count_round_1'].items():
        id = instance['id'] + "-" + label[0]
        if count:
            id_to_negative_label_count[id] =  - count

prediction_file = prediction_dir / "label_count" / "label_count.json"
prediction_file.parent.mkdir(exist_ok=True, parents=True)
with prediction_file.open("w") as f:
    json.dump(id_to_negative_label_count, f)