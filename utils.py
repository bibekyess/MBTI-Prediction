import datetime
import torch
import numpy as np
from sklearn.metrics import roc_auc_score

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)

def convert_mbti_to_label(mbti: str, type: str):
    """
    :param mbti: string. length=4
    :return:
    """
    stand = 'ISTJ'  # [0, 0, 0, 0]
    label_type = {'ie': 0, 'sn': 1, 'tf': 2, 'jp': 3}
    result = []
    for i in range(4):
        if stand[i] == mbti[i]:
            result.append(0)
        else:
            result.append(1)
    return result[label_type.get(type)]

def accuracy_and_auc(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    accuracy = np.sum(pred_flat == labels_flat) / len(labels_flat)

    auc_score = roc_auc_score(labels_flat, preds[:, 1])  # Assuming binary classification
    return accuracy, auc_score

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

