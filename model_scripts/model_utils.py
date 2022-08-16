# Imports
import torch

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# Functions
def tensorify(data, input_type='file'):
    if input_type == 'file':
        dfl = pd.read_csv(data)
        features = dfl.iloc[:, :-1]
        labels = dfl.iloc[:, -1]
        features, labels = torch.tensor(features.values).float(), torch.tensor(labels.values).long()
        # features = features.view(-1, 2, 15)
        features = features.view(-1, 2, features.shape[1] // 2)
        labels = labels.view(-1, 1)
        return features.transpose(2, 1), labels
    elif input_type == 'sample':
        features = torch.tensor(data).float()
        # features = features.view(-1, 2, 15)
        features = features.view(-1, 2, features.shape[1] // 2)
        features = features.transpose(2, 1)
        return features
    else:
        raise ValueError(f'input_type can either be "file", "sample". Cannot be {input_type}')


def validate(model, dataset):
    y_pred = []
    y_true = []
    with torch.no_grad():
        for batchX, batchy in dataset.get_next_batch(use_data='validation'):
            ypred = model(batchX)
            yp = ypred.argmax(axis=1).view(-1).numpy().tolist()
            yt = batchy.view(-1).numpy().tolist()
            y_pred.extend(yp)
            y_true.extend(yt)
    return get_metrics(ytrue=y_true, ypred=y_pred)


def test(model, test_fname):
    features, labels = tensorify(test_fname)
    with torch.no_grad():
        ypred = model(features)
        yp = ypred.argmax(axis=1).view(-1).numpy().tolist()
        yt = labels.view(-1).numpy().tolist()
    return get_metrics(ytrue=yt, ypred=yp)


def get_metrics(ytrue, ypred):
    acc = accuracy_score(y_true=ytrue, y_pred=ypred)
    bacc = balanced_accuracy_score(y_true=ytrue, y_pred=ypred)
    pre = precision_score(y_true=ytrue, y_pred=ypred)
    rec = recall_score(y_true=ytrue, y_pred=ypred)
    f1s = f1_score(y_true=ytrue, y_pred=ypred)
    cfm = confusion_matrix(y_true=ytrue, y_pred=ypred)
    return acc, bacc, pre, rec, f1s, cfm


def predict(model, sample, return_prob=False):
    with torch.no_grad():
        ypred = torch.nn.functional.softmax(model(sample))
    result = ypred[:,1].tolist()
    # result = ypred.view(-1).tolist()[1]
    if return_prob:
        return result
    else:
        return [1 if x >= 0.5 else 0 for x in result]
        # return 1 if result>=0.5 else 0


def process_sample(model, feature_list, return_prob=False):
    return predict(model, tensorify(feature_list, input_type='sample'), return_prob=return_prob)
