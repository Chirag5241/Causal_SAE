
import os
import sys
import random
import torch
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from dataclasses import asdict
from tqdm.auto import tqdm
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, BertForMaskedLM, BertConfig
from torch.optim import Adam
from sklearn.model_selection import ParameterGrid


class MLP(nn.Module):
    def __init__(self, dims=(768, 384, 384, 1)):
        super().__init__()

        layers = []
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:  # Don't add ReLU after last layer
                layers.append(nn.ReLU())
            elif i == len(dims)-2:  # Don't add dropout for second last layer
                layers.append(nn.Dropout(p=0.1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        if len(x.shape) == 4:
            # Flatten the input tensor from [N, 1, 1, 768] to [N, 768]
            x = x.view(x.size(0), -1)
        return self.model(x)


def get_probe_accuracy(logits, labels):
    assert logits.shape[0] == labels.shape[0]
    assert labels.ndim == 2 and labels.shape[1] == 1
    if logits.ndim == 1 or logits.shape[1] == 1:    # binary classification
        assert torch.all(torch.logical_or(labels == 0, labels == 1))
        preds = torch.where(torch.sigmoid(logits) > 0.5, 1, 0)
    else:                                           # multiclass classification
        preds = torch.argmax(logits, axis=1).unsqueeze(1)
    return (preds == labels).sum().float() / labels.shape[0], labels.squeeze(1).tolist()


def df2dataloader(X, y, batch_size=1, shuffle=True, device='cpu'):
    """
    convert dataframe to TensorDataset for use in training probes
    :param df: dataframe with 'mask_embed' and 'label' columns (specify these via the keyword args if different)
    :returns: TensorDataset of concatenated (hyper_embed, mask_embed) with labels {0, 1}
    """
    assert X.shape[0] == y.shape[0]
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_probe(probe, loss_fn, train_loader, val_loader, hypers, device='cpu', do_print=True, do_tqdm=True):
    """
    :param acc_fn: map from 
    :param hypers: dict that must include the following entries: 
        'epochs': int number of training epochs
        'save_criterion': either None (just return params from final epoch), or string in 
            ['train loss', 'train acc', 'val loss', 'val acc']
    :returns: (probe, record), where:
        probe has params from best epoch (as measured by hypers['save_criterion'])
        record has per-epoch loss and accuracy for train and validation sets
    """
    # mlp = MLP().to(device)
    optimizer = Adam(probe.parameters(), lr=hypers['learning_rate'])

    best_state = None
    best_criterion_value = float(
        'inf') if 'loss' in hypers['save_criterion'] else 0
    record = defaultdict(list)

    iteration_range = tqdm(
        range(hypers['epochs'])) if do_tqdm else range(hypers['epochs'])
    for epoch in iteration_range:
        probe.train()
        train_loss, train_acc, train_count = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            y_pred = probe(x)

            if isinstance(loss_fn, nn.CrossEntropyLoss):
                loss = loss_fn(y_pred, y)
            else:
                loss = loss_fn(y_pred, y.unsqueeze(1).float())

            acc, _ = get_probe_accuracy(y_pred, y.unsqueeze(1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += acc.item()
            train_count += 1

        avg_train_loss = train_loss / train_count
        avg_train_acc = train_acc / train_count
        record['train loss'].append(avg_train_loss)
        record['train acc'].append(avg_train_acc)

        avg_val_loss, avg_val_acc, _, _ = test_probe(probe, val_loader, device, loss_fn)
        record['val loss'].append(avg_val_loss)
        record['val acc'].append(avg_val_acc)

        current_criterion_value = record[hypers['save_criterion']][-1]
        if 'acc' in hypers['save_criterion'] and current_criterion_value > best_criterion_value:
            best_state = probe.state_dict()
            best_criterion_value = current_criterion_value
        elif 'loss' in hypers['save_criterion'] and current_criterion_value < best_criterion_value:
            best_state = probe.state_dict()
            best_criterion_value = current_criterion_value

    if best_state:
        probe.load_state_dict(best_state)

    return probe, record


def train_mlp_probe(train_loader, val_loader, hypers, device='cpu', do_print=True, do_tqdm=True):
    mlp = MLP(hypers['dims']).to(device)  # send dims
    loss_fn = hypers["loss_fn"]
    assert isinstance(loss_fn, nn.CrossEntropyLoss)
    return train_probe(mlp, loss_fn, train_loader, val_loader, hypers, device=device, do_print=do_print, do_tqdm=do_tqdm)


def create_dims(n_layers=2, layer_size=384, loss=nn.CrossEntropyLoss()):
    if isinstance(loss, nn.CrossEntropyLoss):
        dims = [768] + [layer_size] * n_layers + \
            [2]  # need to make it flexible CHIRAG
    else:
        dims = [768] + [layer_size] * n_layers + [1]
    return dims


def find_best_probe(X, y, X_test, y_test, param_grid, batch_size, criterion='val acc', do_print=True, do_tqdm=True):
    assert criterion in {'train acc', 'train loss', 'val acc', 'val loss'}
    assert X.shape[0] == y.shape[0]
    assert (X.shape[1] == 768 or X.shape[3] == 768) and (
        len(X.shape) == 2 or len(X.shape) == 4) and len(y.shape) == 1
    assert X.shape[1] == X_test.shape[1]
    assert len(y_test.shape) == 1

    X_train, X_test, y_train, y_test = X, X_test, y, y_test
    print("Train, test num. examples:",
          X_train.shape[0], X_test.shape[0])

    train_loader = df2dataloader(X_train, y_train, batch_size=batch_size)
    test_loader = df2dataloader(X_test, y_test, batch_size=batch_size, shuffle = False)

    best_criterion, best_params, best_probe = None, None, None

    for params in ParameterGrid(param_grid):
        hypers = {
            'n_layers': params['n_layers'],
            'layer_size': params['layer_size'],
            'dims': create_dims(params['n_layers'], params['layer_size'], params['loss_fn']),
            'epochs': 8,
            'save_criterion': criterion,
            'loss_fn': params['loss_fn'],
            'learning_rate': params['learning_rate']
        }

        if do_print:
            print(hypers)
        probe, record = train_mlp_probe(
            train_loader, test_loader, hypers, do_print=do_print, device=X.device, do_tqdm=do_tqdm)
        probe = probe.to(X.device)

        if 'acc' in criterion:
            curr_criterion = max(record[criterion])
        elif 'loss' in criterion:
            curr_criterion = min(record[criterion])
        else:
            raise ValueError(f"{criterion} is not a valid criterion.")
        if do_print:
            print(criterion, curr_criterion)

        if best_criterion is None:
            best_criterion, best_params, best_probe = curr_criterion, hypers, probe
        else:
            if 'acc' in criterion:
                if curr_criterion >= best_criterion:
                    best_criterion, best_params, best_probe = curr_criterion, hypers, probe
            elif 'loss' in criterion:
                if curr_criterion <= best_criterion:
                    best_criterion, best_params, best_probe = curr_criterion, hypers, probe
            else:
                raise ValueError(f"{criterion} is not a valid criterion.")

    print(f"Best {criterion}: {best_criterion} with parameters {best_params}")
    return best_probe, best_criterion



def create_linear_dims(output_classes=2):
    # Always 768 for input (as BERT-like models output) and output_classes depending on the task (binary or multi-class)
    return [768, output_classes]

def train_linear_probe(train_loader, val_loader, hypers, device='cpu', do_print=True, do_tqdm=True):
    linear_model = nn.Linear(hypers['dims'][0], hypers['dims'][1]).to(device)
    loss_fn = hypers["loss_fn"]
    assert isinstance(loss_fn, nn.CrossEntropyLoss)
    return train_probe(linear_model, loss_fn, train_loader, val_loader, hypers, device=device, do_print=do_print, do_tqdm=do_tqdm)

def df2dataloader(X, y, batch_size, shuffle=True):
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def find_best_linear_probe(X, y, X_test, y_test, param_grid, batch_size, criterion='val acc', do_print=True, do_tqdm=True):
    assert criterion in {'train acc', 'train loss', 'val acc', 'val loss'}
    assert X.shape[0] == y.shape[0] and X.shape[1] == X_test.shape[1] and len(y.shape) == 1

    train_loader = df2dataloader(X, y, batch_size=batch_size)
    test_loader = df2dataloader(X_test, y_test, batch_size=batch_size, shuffle=False)

    best_criterion, best_params, best_probe = None, None, None

    for params in ParameterGrid(param_grid):
        hypers = {
            'dims': create_linear_dims(params['output_classes']),
            'epochs': 20,
            'save_criterion': criterion,
            'loss_fn': params['loss_fn'],
            'learning_rate': params['learning_rate']
        }

        if do_print:
            print(f"Hyperparameters: {hypers}")
        probe, record = train_linear_probe(train_loader, test_loader, hypers, device=X.device, do_print=do_print, do_tqdm=do_tqdm)

        if 'acc' in criterion:
            curr_criterion = max(record[criterion])
        elif 'loss' in criterion:
            curr_criterion = min(record[criterion])

        if do_print:
            print(f"{criterion}: {curr_criterion}")

        if best_criterion is None or ('acc' in criterion and curr_criterion >= best_criterion) or ('loss' in criterion and curr_criterion <= best_criterion):
            best_criterion, best_params, best_probe = curr_criterion, hypers, probe

    print(f"Best {criterion}: {best_criterion} with parameters {best_params}")
    return best_probe, best_criterion




def feed_test_set_thru_probe(probe, X, y, batch_size, compute_all_preds=False, expander=None):
    assert X.shape[0] == y.shape[0]
    if expander is None:
        test_loader = df2dataloader(X, y, batch_size=batch_size, shuffle = False)
        test_loss, test_acc, logits, labs = test_probe(probe, test_loader, X.device, nn.CrossEntropyLoss(), compute_all_preds=compute_all_preds)
        if compute_all_preds:
            assert logits.shape[0] == X.shape[0]
    else:
        assert False, 'havent looked at in a while.'
        N = len(expander)
        test_loss, test_acc = (0,0), (0,0)
        logits = None  # for nw
        for i in range(0, N, batch_size):
            expander_here = expander[i:i+batch_size]
            Xh, yh = X[expander_here], y[expander_here]
            test_loader = df2dataloader(Xh, yh, batch_size=batch_size, shuffle=False)
            test_loss_h, test_acc_h, logits_h, labs = test_probe(probe, test_loader, X.device, nn.CrossEntropyLoss(),
                                                     compute_all_preds=compute_all_preds, normalize_scores=False)
            assert logits_h is None, 'FIX WHEN COMES UP'
            test_loss = (test_loss[0] + test_loss_h[0], test_loss[1] + test_loss_h[1])
            test_acc = (test_acc[0] + test_acc_h[0], test_acc[1] + test_acc_h[1])
        test_loss = test_loss[0] / test_loss[1]
        test_acc = test_acc[0] / test_acc[1]
    return test_loss, test_acc, logits, labs


def test_probe(probe, test_loader, device, loss_fn, compute_all_preds = False, normalize_scores = True):
    probe.eval()
    val_loss, val_acc, val_count = 0, 0, 0
    if compute_all_preds: all_logits = []
    labs = []
    with torch.no_grad():
        for x_val, y_val in test_loader:
            x_val, y_val = x_val.to(device), y_val.to(device)
            y_pred_val = probe(x_val)
            if compute_all_preds: all_logits.append(y_pred_val)
            if isinstance(loss_fn, nn.CrossEntropyLoss):
                loss = loss_fn(y_pred_val, y_val)
            else:
                loss = loss_fn(y_pred_val, y_val.unsqueeze(1).float())

            acc, labsh = get_probe_accuracy(y_pred_val, y_val.unsqueeze(1))
            labs += labsh

            val_loss += loss.item()
            val_acc += acc.item()
            val_count += 1

    avg_val_loss = (val_loss / val_count) if normalize_scores else (val_loss, val_count)
    avg_val_acc = (val_acc / val_count) if normalize_scores else (val_acc, val_count)
    return avg_val_loss, avg_val_acc, torch.cat(all_logits, dim=0) if compute_all_preds else None, labs


def get_control_probe_labels(data):
    """
    Randomly assign half of verb lemmas to group A or B:
        A: for each instance, set the y label equal to the correct value of “is_singular” 
            (e.g., “smiles” is still singular, “smile” is still plural)
        B: flip the y label from the original value of “is_singular” 
            (e.g., “smiles” is now plural, “smile” is now singular)
    :param data: flattened data
    :returns: control probe labels, as described above
    """
    lemmas = set([d['trg1_wd'] if d['is_singular'] else d['trg1_fl'] for d in data])
    flipped = set(random.sample(list(lemmas), k=len(lemmas) // 2))
    # not_flipped = lemmas - flipped
    
    labels = list()
    for d in data:
        if d['is_singular']:                            # would normally get y = 1
            y = 0 if d['trg1_wd'] in flipped else 1     # change to y = 0 if we're flipping this lemma
        else:                                           # would normally get y = 0
            y = 1 if d['trg1_wd'] in flipped else 0     # change to y = 1 if we're flipping this lemma
        labels.append(y)

    return torch.tensor(labels)
