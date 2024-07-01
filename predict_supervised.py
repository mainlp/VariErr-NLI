import argparse

import numpy as np
import torch
import datasets
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from predict_scorers import get_probs_per_epoch, write_results, LABEL_TO_ID

def get_is_label_error(dataset: datasets.Dataset) -> np.ndarray:
    y = np.zeros((len(dataset), 3))
    for idx_instance, instance in enumerate(dataset):
        for error_label in instance['error_labels']:
            y[idx_instance][LABEL_TO_ID[error_label]] = 1

    return y
    


def predict_metadata_archaeology(outputs: torch.Tensor, dataset: datasets.Dataset, clf) -> np.ndarray:
    preds = np.zeros((len(dataset), 3))
    probs_per_epoch = np.array(get_probs_per_epoch(outputs))
    loss_per_epoch = -np.log(probs_per_epoch)
    is_label_error = get_is_label_error(dataset)
    num_features = loss_per_epoch.shape[0]
    idcs = np.arange(len(dataset))

    for idcs_train, idcs_test in KFold(n_splits=2, shuffle=True).split(idcs):
        X_train = loss_per_epoch[:, idcs_train].transpose(1, 2, 0).reshape(-1, num_features)
        y_train = is_label_error[idcs_train].reshape(-1)

        # assert(y_train.reshape(-1, 3) == is_label_error[idcs_train]).all()

        # for i in range(100):
        #     for j in range(3):
        #         assert all(X_train[i * 3 + j] == loss_per_epoch[:, idcs_train][:, i, j])
        #         assert y_train[i * 3 + j] == is_label_error[idcs_train][i, j]
        
        clf = KNeighborsClassifier(n_neighbors=20)
        # clf = LogisticRegression()
        clf.fit(X_train, y_train)
        X_test = loss_per_epoch[:, idcs_test].transpose(1, 2, 0).reshape((-1, num_features))
        y_test_pred = clf.predict_proba(X_test)[:, 1]

        preds[idcs_test] = y_test_pred.reshape(-1, 3)

    return preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs', type=Path, required=True)
    args = parser.parse_args()

    output_dir = Path("predictions")


    dataset = datasets.load_dataset("mainlp/varierr")['train']
    outputs = torch.load(str(args.outputs))
    model_name = args.outputs.parent.name

    preds_metadata_archaeology_knn = predict_metadata_archaeology(outputs=outputs, dataset=dataset, clf=KNeighborsClassifier(n_neighbors=20))
    write_results(predictions=preds_metadata_archaeology_knn, dataset=dataset, output_path=output_dir / 'metadata_archaeology_knn' / model_name)

    preds_metadata_archaeology_lr = predict_metadata_archaeology(outputs=outputs, dataset=dataset, clf=LogisticRegression())
    write_results(predictions=preds_metadata_archaeology_lr, dataset=dataset, output_path=output_dir / 'metadata_archaeology_lr' / model_name)


