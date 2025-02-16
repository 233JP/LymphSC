import argparse
import torch
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, KFold
from dataset_graph import construct_dataset, mol_collate_func
from transformer_graph_finetune import make_model
from utils import ScheduledOptim, get_options, get_loss, cal_loss, evaluate, scaffold_split
from collections import defaultdict
from Model import GATNet
import matplotlib.pyplot as plt


def model_train(model, train_dataset, valid_dataset, model_params, train_params, dataset_name, experiment_name,fold):
    # build data loader
    train_loader = DataLoader(dataset=train_dataset, batch_size=train_params['batch_size'], collate_fn=mol_collate_func,
                              shuffle=True, drop_last=True, num_workers=4, pin_memory=True)

    valid_loader = DataLoader(dataset=valid_dataset, batch_size=train_params['batch_size'], collate_fn=mol_collate_func,
                              shuffle=True, drop_last=True, num_workers=4, pin_memory=True)

    # build loss function
    criterion = get_loss(train_params['loss_function'])

    # build optimizer
    optimizer = ScheduledOptim(torch.optim.Adam(model.parameters(), lr=0),
                               train_params['warmup_factor'], model_params['d_model'],
                               train_params['total_warmup_epochs'])

    best_valid_metric = float('inf') if train_params['task'] == 'regression' else float('-inf')
    best_epoch = -1
    best_valid_result, best_valid_bedding = None, None

    # 用于存储损失和准确率
    train_losses = []
    vaild_accuracies = []

    for epoch in range(train_params['total_epochs']):
        # train
        train_loss = list()
        model.train()
        for batch in tqdm(train_loader):
            smile_list, adjacency_matrix, node_features, edge_features, y_true = batch
            adjacency_matrix = adjacency_matrix.to(train_params['device'])  # (batch, max_length, max_length)
            node_features = node_features.to(train_params['device'])  # (batch, max_length, d_node)
            edge_features = edge_features.to(train_params['device'])  # (batch, max_length, max_length, d_edge)
            y_true = y_true.to(train_params['device'])  # (batch, task_numbers)
            batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0  # (batch, max_length)
            # (batch, task_numbers)
            y_pred, _ = model(node_features, batch_mask, adjacency_matrix, edge_features)
            loss = cal_loss(y_true, y_pred, train_params['loss_function'], criterion,
                            0, 1, train_params['device'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step_and_update_lr()
            train_loss.append(loss.detach().item())
        train_losses.append(np.mean(train_loss))
        # valid
        model.eval()
        with torch.no_grad():
            valid_true, valid_pred, valid_smile, valid_embedding = list(), list(), list(), list()
            for batch in tqdm(valid_loader):
                smile_list, adjacency_matrix, node_features, edge_features, y_true = batch
                adjacency_matrix = adjacency_matrix.to(train_params['device'])  # (batch, max_length, max_length)
                node_features = node_features.to(train_params['device'])  # (batch, max_length, d_node)
                edge_features = edge_features.to(train_params['device'])  # (batch, max_length, max_length, d_edge)
                batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0  # (batch, max_length)
                # (batch, task_numbers)
                y_pred, y_embedding = model(node_features, batch_mask, adjacency_matrix, edge_features)

                y_true = y_true.numpy()  # (batch, task_numbers)
                y_pred = y_pred.detach().cpu().numpy()  # (batch, task_numbers)
                y_embedding = y_embedding.detach().cpu().numpy()

                valid_true.append(y_true)
                valid_pred.append(y_pred)
                valid_smile.append(smile_list)
                valid_embedding.append(y_embedding)

            valid_true, valid_pred = np.concatenate(valid_true, axis=0), np.concatenate(valid_pred, axis=0)
            valid_smile, valid_embedding = np.concatenate(valid_smile, axis=0), np.concatenate(valid_embedding, axis=0)

        valid_result = evaluate(valid_true, valid_pred, valid_smile,
                                requirement=['sample', train_params['loss_function'], train_params['metric']],
                                data_mean=0, data_std=1, data_task=train_params['task'])
        # 记录测试准确率
        vaild_accuracy = valid_result[train_params['metric']]
        vaild_accuracies.append(vaild_accuracy)

        if valid_result[train_params['metric']] > best_valid_metric:
            best_valid_metric = valid_result[train_params['metric']]
            best_epoch = epoch + 1
            best_valid_result = valid_result
            best_valid_bedding = valid_embedding
            torch.save({'state_dict': model.state_dict(),
                        'best_epoch': best_epoch,
                        f'best_valid_{train_params["metric"]}': best_valid_metric},
                       f'./../Model/{dataset_name}/best_model_{experiment_name}_{dataset_name}_fold_{fold}.pt')

        print("Epoch {}, learning rate {:.6f}, "
              "train {}: {:.4f}, "
              "valid {}: {:.4f}, "
              "valid {}: {:.4f}, "
              "valid precision:{:.4f}"
              "best epoch {}, best valid {}: {:.4f}"
              .format(epoch + 1, optimizer.view_lr(),
                      train_params['loss_function'], np.mean(train_loss),
                      train_params['loss_function'], valid_result[train_params['loss_function']],
                      train_params['metric'], valid_result[train_params['metric']],
                      valid_result['precision'],
                      best_epoch, train_params['metric'], best_valid_metric
                      ))

        # early stop
        if abs(best_epoch - epoch) >= 20:
            print("=" * 20 + ' early stop ' + "=" * 20)
            break

    return best_valid_result, best_valid_bedding, train_losses, vaild_accuracies


def model_test(checkpoint, test_dataset, model_params, train_params):
    # build loader
    test_loader = DataLoader(dataset=test_dataset, batch_size=train_params['batch_size'], collate_fn=mol_collate_func,
                             shuffle=False, drop_last=True, num_workers=4, pin_memory=True)

    # build model
    model = make_model(**model_params)
    model.to(train_params['device'])
    model.load_state_dict(checkpoint['state_dict'])
    # model.load_state_dict(torch.load('./../Model/lymph/best_model_None_linba_fold_1.pt'))
    test_accuracies = []
    # test
    model.eval()
    with torch.no_grad():
        test_true, test_pred, test_smile, test_embedding = list(), list(), list(), list()
        for batch in tqdm(test_loader):
            smile_list, adjacency_matrix, node_features, edge_features, y_true = batch
            adjacency_matrix = adjacency_matrix.to(train_params['device'])  # (batch, max_length, max_length)
            node_features = node_features.to(train_params['device'])  # (batch, max_length, d_node)
            edge_features = edge_features.to(train_params['device'])  # (batch, max_length, max_length, d_edge)
            batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0  # (batch, max_length)
            # (batch, task_numbers)
            y_pred, y_embedding = model(node_features, batch_mask, adjacency_matrix, edge_features)

            y_true = y_true.numpy()  # (batch, task_numbers)
            y_pred = y_pred.detach().cpu().numpy()  # (batch, task_numbers)
            y_embedding = y_embedding.detach().cpu().numpy()

            test_true.append(y_true)
            test_pred.append(y_pred)
            test_smile.append(smile_list)
            test_embedding.append(y_embedding)
        print(test_true)
        print(test_pred)
        test_true, test_pred = np.concatenate(test_true, axis=0), np.concatenate(test_pred, axis=0)
        test_smile, test_embedding = np.concatenate(test_smile, axis=0), np.concatenate(test_embedding, axis=0)
    test_result = evaluate(test_true, test_pred, test_smile,
                           requirement=['sample', train_params['loss_function'], train_params['metric']],
                           data_mean=0, data_std=1, data_task=train_params['task'])

    print("test {}: {:.4f}".format(train_params['metric'], test_result[train_params['metric']]))
    # test_accuracy = test_result[train_params['metric']]
    # test_accuracies.append(test_accuracy)
    return test_result, test_embedding


if __name__ == '__main__':
    # init args
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="random seeds", default=np.random.randint(1000))
    parser.add_argument("--gpu", type=str, help='gpu', default=0)
    parser.add_argument("--fold", type=int, help='the number of k-fold', default=1)
    parser.add_argument('--model', help="name of models to be used")
    parser.add_argument("--dataset", type=str, help='choose a dataset', default='lymph')
    parser.add_argument("--split", type=str, help="choose the split type", default='scaffold', choices=['random', 'scaffold'])
    parser.add_argument("--experiment_name", type=str, help="experiment_name", default=None)
    parser.add_argument('--train', action='store_true', help='Train phase over the full training datasets')

    args = parser.parse_args()
    is_training = args.train
    model_name = args.model

    # load options
    model_params, train_params = get_options(args.dataset)

    if torch.cuda.is_available():
        train_params['device'] = torch.device(f'cuda:{args.gpu}')
        torch.cuda.manual_seed(args.seed)
    else:
        train_params['device'] = torch.device('cpu')

    # load data

    with open(f'./../Data/Data/{args.dataset}/preprocess/{args.dataset}.pickle', 'rb') as f:
        [data_mol, data_label] = pkl.load(f)

    # calculate the padding
    model_params['max_length'] = max([data.GetNumAtoms() for data in data_mol])
    print(f"Max padding length is: {model_params['max_length']}")

    # construct dataset
    print('=' * 20 + ' construct dataset ' + '=' * 20)
    dataset = construct_dataset(data_mol, data_label, model_params['d_atom'], model_params['d_edge'], model_params['max_length'])
    total_metrics = defaultdict(list)

    # split dataset
    if args.split == 'scaffold':
        # we run the scaffold split 5 times for different random seed, which means different train/valid/test
        for idx in range(args.fold):
            print('=' * 20 + f' train on fold {idx + 1} ' + '=' * 20)
            print(f"Seed: {args.seed+ idx}")
            np.random.seed(args.seed+idx)
            torch.manual_seed(args.seed+idx)
            if torch.cuda.is_available():
                train_params['device'] = torch.device(f'cuda:{args.gpu}')
                torch.cuda.manual_seed(args.seed+idx)
            # get dataset
            train_index, valid_index, test_index = scaffold_split(data_mol, frac=[0.7, 0.2, 0.1], balanced=True,
                                                                  include_chirality=False, ramdom_state=args.seed + idx)
            train_dataset, valid_dataset, test_dataset = dataset[train_index], dataset[valid_index], dataset[test_index]

            # calculate total warmup steps
            train_params['total_warmup_steps'] = \
                int(len(train_dataset) / train_params['batch_size']) * train_params['total_warmup_epochs']
            print('train warmup step is: {}'.format(train_params['total_warmup_steps']))

            train_params['mean'], train_params['std'] = 0, 1

            model = make_model(**model_params)
            model = model.to(train_params['device'])

            # train and valid
            print(f"train size: {len(train_dataset)}, valid size: {len(valid_dataset)}, test size: {len(test_dataset)}")
            best_valid_result, _, train_losses = model_train(model, train_dataset, valid_dataset, model_params, train_params, args.dataset, args.experiment_name,idx + 1)
            best_valid_csv = pd.DataFrame.from_dict({'smile': best_valid_result['smile'], 'actual': best_valid_result['label'], 'predict': best_valid_result['prediction']})
            best_valid_csv.to_csv(f'./Result/{args.dataset}/best_valid_result_{args.experiment_name}_{args.dataset}_fold_{idx + 1}.csv', sep=',', index=False, encoding='UTF-8')
            total_metrics['valid'].append(best_valid_result[train_params['metric']])
            # test
            print('=' * 20 + f' test on fold {idx + 1} ' + '=' * 20)
            checkpoint = torch.load(f'./Model/{args.dataset}/best_model_{args.experiment_name}_{args.dataset}_fold_{idx + 1}.pt', map_location=train_params['device'])
            test_result, test_embedding,  = model_test(checkpoint, test_dataset, model_params, train_params)
            test_csv = pd.DataFrame.from_dict({'smile': test_result['smile'], 'actual': test_result['label'], 'predict': test_result['prediction']})
            test_csv.to_csv(f'./Result/{args.dataset}/best_test_result_{args.experiment_name}_{args.dataset}_fold_{idx + 1}.csv', sep=',', index=False, encoding='UTF-8')
            total_metrics['test'].append(test_result[train_params['metric']])

            total_embedding = dict()
            for smile, embedding in zip(test_result['smile'], test_embedding):
                total_embedding[smile] = embedding

            with open(f'./Result/{args.dataset}/total_test_embedding_fold_{idx + 1}.pickle', 'wb') as fw:
                pkl.dump(total_embedding, fw)

        print('=' * 20 + ' summary ' + '=' * 20)
        print('Seed: {}'.format(args.seed))
        for idx in range(args.fold):
            print('fold {}, valid {} = {:.4f}, test {} = {:.4f}'
                  .format(idx + 1,
                          train_params['metric'], total_metrics['valid'][idx],
                          train_params['metric'], total_metrics['test'][idx]))

        print('{} folds valid average {} = {:.4f} ± {:.4f}, test average {} = {:.4f} ± {:.4f}'
              .format(args.fold,
                      train_params['metric'], np.nanmean(total_metrics['valid']), np.nanstd(total_metrics['valid']),
                      train_params['metric'], np.nanmean(total_metrics['test']), np.nanstd(total_metrics['test']),
                      ))
        print('=' * 20 + " finished! " + '=' * 20)

    # else args.split == 'random':
    else:
        # we run the random split 5 times for different random seed, which means different train/valid/test
        for idx in range(args.fold):
            print('=' * 20 + f' train on fold {idx + 1} ' + '=' * 20)
            # print('=' * 20 + f' train on fold {idx + 1} ' + '=' * 20)
            print(f"Seed: {args.seed + idx}")
            np.random.seed(args.seed + idx)
            torch.manual_seed(args.seed + idx)
            if torch.cuda.is_available():
                train_params['device'] = torch.device(f'cuda:{args.gpu}')
                torch.cuda.manual_seed(args.seed + idx)

            # define a model
            model = make_model(**model_params)
            model = model.to(train_params['device'])

            # get dataset
            if is_training:
                train_valid_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=args.seed)
                train_dataset, valid_dataset = train_test_split(train_valid_dataset, test_size=len(test_dataset),
                                                                random_state=args.seed)

                # train and valid
                print(f"train size: {len(train_dataset)}, valid size: {len(valid_dataset)}, test size: {len(test_dataset)}")
                best_valid_result, _, train_losses, vaild_accuracies = model_train(model, train_dataset, valid_dataset, model_params, train_params, args.dataset, args.experiment_name, idx + 1)
                best_valid_csv = pd.DataFrame.from_dict({'actual': best_valid_result['label'], 'predict': best_valid_result['prediction']})
                best_valid_csv.to_csv(f'./../Result/{args.dataset}/best_valid_result_{args.experiment_name}_{args.dataset}_fold_{idx + 1}.csv', sep=',', index=False, encoding='UTF-8')
                total_metrics['valid'].append(best_valid_result[train_params['metric']])

                # test
                print('=' * 20 + f' test on fold {idx + 1} ' + '=' * 20)
                checkpoint = torch.load(f'./../Model/{args.dataset}/best_model_{args.experiment_name}_{args.dataset}_fold_1.pt')
                test_result, _ = model_test(checkpoint, test_dataset, model_params, train_params)
                test_csv = pd.DataFrame.from_dict({'actual': test_result['label'], 'predict': test_result['prediction']})
                test_csv.to_csv(f'./../MyResult/{args.dataset}/best_test_result_{args.experiment_name}_{args.dataset}_fold_3.csv', sep=',', index=False, encoding='UTF-8')
                # test_csv.to_csv(f'./../Result/{args.dataset}/best_test_result_{args.experiment_name}_{args.dataset}_fold_{idx + 1}.csv', sep=',', index=False, encoding='UTF-8')
                total_metrics['test'].append(test_result[train_params['metric']])

            else:
                test_dataset = dataset
                # test
                print('=' * 20 + f' test on fold {idx + 1} ' + '=' * 20)
                checkpoint = torch.load(f'./../Model/{args.dataset}/best_model_{args.experiment_name}_{args.dataset}_fold_1.pt')
                test_result, _= model_test(checkpoint, test_dataset, model_params, train_params)
                test_csv = pd.DataFrame.from_dict({'actual': test_result['label'], 'predict': test_result['prediction']})
                test_csv.to_csv(f'./../MyResult/{args.dataset}/best_test_result_{args.experiment_name}_{args.dataset}_fold_3.csv', sep=',', index=False, encoding='UTF-8')
                # test_csv.to_csv(f'./../Result/{args.dataset}/best_test_result_{args.experiment_name}_{args.dataset}_fold_{idx + 1}.csv', sep=',', index=False, encoding='UTF-8')
                total_metrics['test'].append(test_result[train_params['metric']])

        print('=' * 20 + ' summary ' + '=' * 20)



