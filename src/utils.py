import math
import torch
from random import Random
from torch.nn.init import _calculate_fan_in_and_fan_out, _no_grad_normal_, _no_grad_uniform_
from sklearn import metrics
from collections import defaultdict
import numpy as np
import torch.nn.functional as F
from rdkit.Chem.Scaffolds import MurckoScaffold


def xavier_normal_small_init_(tensor, gain=1.):
    # type: (Tensor, float) -> Tensor
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + 4 * fan_out))

    return _no_grad_normal_(tensor, 0., std)


def xavier_uniform_small_init_(tensor, gain=1.):
    # type: (Tensor, float) -> Tensor
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + 4 * fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

    return _no_grad_uniform_(tensor, -a, a)


def get_loss(loss_name):
    if loss_name == 'rmse':
        return torch.nn.MSELoss()
    elif loss_name == 'mae':
        return torch.nn.L1Loss()
    elif loss_name == 'bce':
        return torch.nn.BCEWithLogitsLoss(reduction='none')
    elif loss_name == 'ce':
        return torch.nn.CrossEntropyLoss(ignore_index=-1)
    else:
        assert 'Loss function not supported !'


def cal_loss(y_true, y_pred, loss_name, criterion, data_mean, data_std, device):
    # y_true, y_pred.shape = (batch, num_tasks)
    if loss_name == 'rmse':
        y_true, y_pred = (y_true.flatten() - data_mean)/data_std, y_pred.flatten()
        loss = torch.sqrt(criterion(y_pred, y_true))
    elif loss_name == 'mae':
        y_true, y_pred = (y_true.flatten() - data_mean)/data_std, y_pred.flatten()
        loss = criterion(y_pred, y_true)
    elif loss_name == 'bce':
        # find all -1 in y_true
        y_true = y_true.long()
        y_mask = torch.where(y_true == -1, torch.tensor([0]).to(device), torch.tensor([1]).to(device))
        y_cal_true = torch.where(y_true == -1, torch.tensor([0]).to(device), y_true).float()
        loss = criterion(y_pred, y_cal_true) * y_mask
        loss = loss.sum() / y_mask.sum()
    else:
        loss = criterion(y_pred, y_true)
    return loss


def evaluate(y_true, y_pred, y_smile, requirement, data_mean, data_std, data_task):
    # y_true.shape = y_pred.shape = (samples, task_numbers)
    print(y_true)
    print(y_pred)
    collect_result = {}
    if 'sample' in requirement:
        collect_result['smile'] = y_smile.tolist()
        if data_task == 'classification':
            collect_result['prediction'] = y_pred.tolist()
            collect_result['label'] = y_true.tolist()
        else:
            collect_result['prediction'] = (y_pred * data_std + data_mean).tolist()
            collect_result['label'] = y_true.tolist()
    if 'rmse' in requirement:
        y_true, y_pred = y_true.flatten(), (y_pred.flatten() * data_std + data_mean).tolist()
        collect_result['rmse'] = np.sqrt(F.mse_loss(torch.tensor(y_pred), torch.tensor(y_true), reduction='mean'))
    if 'mae' in requirement:
        y_true, y_pred = y_true.flatten(), (y_pred.flatten() * data_std + data_mean).tolist()
        collect_result['mae'] = F.l1_loss(torch.tensor(y_pred), torch.tensor(y_true), reduction='mean')
    if 'bce' in requirement:
        # find all -1 in y_true
        y_mask = np.where(y_true == -1, 0, 1)
        y_cal_true = np.where(y_true == -1, 0, y_true)
        loss = F.binary_cross_entropy_with_logits(torch.tensor(y_pred), torch.tensor(y_cal_true), reduction='none') * y_mask
        collect_result['bce'] = loss.sum() / y_mask.sum()
    if 'auc' in requirement:
        auc_score_list = []
        if y_true.shape[1] > 1:
            for label in range(y_true.shape[1]):
                true, pred = y_true[:, label], y_pred[:, label]
                # all 0's or all 1's
                if len(set(true)) == 1:
                    auc_score_list.append(float('nan'))
                else:
                    if len(set(true[np.where(true >= 0)])) !=1:
                        auc_score_list.append(metrics.roc_auc_score(true[np.where(true >= 0)], pred[np.where(true >= 0)]))
                    else:
                        auc_score_list.append(float('nan'))
            collect_result['auc'] = np.nanmean(auc_score_list)
        else:
            if len(set(y_true.reshape(-1))) == 1:
                collect_result['auc']=float('nan')
            else:
                threshold = 0
                pre_label = (y_pred > threshold).astype(int)
                y_true_1 = y_true.ravel()
                pre_label_1 = pre_label.ravel()
                collect_result['precision'] = metrics.precision_score(y_true_1, pre_label_1)
                collect_result['auc'] = metrics.roc_auc_score(y_true, y_pred)
    return collect_result


def scaffold_split(mol_list, frac=None, balanced=False, include_chirality=False, ramdom_state=0):
    if frac is None:
        frac = [0.8, 0.1, 0.1]
    # assert sum(frac) == 1

    n_total_valid = int(np.floor(frac[1] * len(mol_list)))
    n_total_test = int(np.floor(frac[2] * len(mol_list)))
    n_total_train = len(mol_list) - n_total_valid - n_total_test

    scaffolds_sets = defaultdict(list)
    for idx, mol in enumerate(mol_list):
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
        scaffolds_sets[scaffold].append(idx)

    random = Random(ramdom_state)

    # Put stuff that's bigger than half the val/test size into train, rest just order randomly
    if balanced:
        index_sets = list(scaffolds_sets.values())
        big_index_sets, small_index_sets = list(), list()
        for index_set in index_sets:
            if len(index_set) > n_total_valid / 2 or len(index_set) > n_total_test / 2:
                big_index_sets.append(index_set)
            else:
                small_index_sets.append(index_set)

        random.seed(ramdom_state)
        random.shuffle(big_index_sets)
        random.shuffle(small_index_sets)
        index_sets = big_index_sets + small_index_sets
    else:  # Sort from largest to smallest scaffold sets
        index_sets = sorted(list(scaffolds_sets.values()), key=lambda index_set: len(index_set), reverse=True)

    train_index, valid_index, test_index = list(), list(), list()
    for index_set in index_sets:
        if len(train_index) + len(index_set) <= n_total_train:
            train_index += index_set
        elif len(valid_index) + len(index_set) <= n_total_valid:
            valid_index += index_set
        else:
            test_index += index_set

    return train_index, valid_index, test_index


class ScheduledOptim:
    """ https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Optim.py """

    def __init__(self, optimizer, factor, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.factor = factor
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0

    def step_and_update_lr(self):
        """Step with the inner optimizer"""
        self._update_learning_rate()
        self._optimizer.step()

    def view_lr(self):
        return self._optimizer.param_groups[0]['lr']

    def zero_grad(self):
        """Zero out the gradients with the inner optimizer"""
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))

    def _update_learning_rate(self):
        """ Learning rate scheduling per step """

        self.n_steps += 1
        lr = self.factor * self._get_lr_scale()
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


def get_options(dataset_name):
    # graph regression

    if dataset_name == 'lymph':
        # model parameters
        model_params = {
            'd_atom': 115,                          # atom features
            'd_edge': 13,                           # edge features
            'd_model': 256,                         # hidden layer
            'N': 4,                                 # Encoder layers
            'h': 4,                                 # Multi-attention heads
            'N_dense': 1,                           # PositionWiseFeedForward layers
            'n_generator_layers': 2,                # Generator layers
            'n_output': 1,
            'leaky_relu_slope': 0.1,
            'dense_output_nonlinearity': 'mish',
            'distance_matrix_kernel': 'exp',
            'aggregation_type': 'gru',
            'scale_norm': False,
            'dropout': 0.0,
            'num_labels': 1,
        }
        # train parameters
        train_params = {
            'total_epochs': 30,
            'batch_size': 25,
            'warmup_factor': 0.01,
            'total_warmup_epochs':6,
            'loss_function': 'bce',
            'metric': 'auc',
            'task': 'classification'
        }
    elif dataset_name == 'lymph_test':
        # model parameters
        model_params = {
            'd_atom': 115,                          # atom features
            'd_edge': 13,                           # edge features
            'd_model': 256,                         # hidden layer
            'N': 4,                                 # Encoder layers
            'h': 4,                                 # Multi-attention heads
            'N_dense': 1,                           # PositionWiseFeedForward layers
            'n_generator_layers': 2,                # Generator layers
            'n_output': 1,
            'leaky_relu_slope': 0.1,
            'dense_output_nonlinearity': 'mish',
            'distance_matrix_kernel': 'exp',
            'aggregation_type': 'gru',
            'scale_norm': False,
            'dropout': 0.0,
            'num_labels': 1,
        }
        # train parameters
        train_params = {
            'total_epochs': 30,
            'batch_size': 14,
            'warmup_factor': 0.01,
            'total_warmup_epochs':6,
            'loss_function': 'bce',
            'metric': 'auc',
            'task': 'classification'
        }
    else:
        pass
    return model_params, train_params




