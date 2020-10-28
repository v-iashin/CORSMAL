import argparse
import ast
import copy
import os
import pathlib
import random
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from time import localtime, strftime
import numpy as np
import pandas as pd
import torch
from model import RNN
from torch.utils.data import DataLoader
from dataset import AudioDataset


def get_cmd_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser('Experiment')
    parser.add_argument('--task', default='flvl')
    parser.add_argument('--output_dim', default=3, type=int)
    parser.add_argument('--model_type', default='GRU')
    parser.add_argument('--bi_dir', dest='bi_dir', action='store_true', default=False)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--data_root', default='./vggish_features')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--input_dim', default=128, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--n_layers', default=5, type=int)
    parser.add_argument('--drop_p', default=0.0, type=float)
    parser.add_argument('--num_epochs', default=5, type=int)
    parser.add_argument('--seed', default=1337, type=int)
    parser.add_argument('--predict_on_private', dest='predict_on_private', action='store_true', default=False)
    args = parser.parse_args()
    return args


class Config(object):

    def __init__(self) -> None:
        '''Generic Config object which adapts to any number of arguments.'''
        self.init_time = strftime('%y%m%d%H%M%S', localtime())

    def assign_variable(self, var_name: str, var_value: str) -> None:
        vars(self)[var_name] = var_value

    def save_self_to(self, path) -> None:
        '''Saves itself as a text file'''
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as fwrite:
            for key, val in vars(self).items():
                fwrite.write(f'{key}: {val}\n')

    def guess_type(self, s):
        try:
            value = ast.literal_eval(s)
            if value is False:
                return False
        except ValueError:
            return str(s)
        except SyntaxError:
            return str(s)
        else:
            return type(value)(s)

    def load_from(self, cmd_args: argparse.Namespace = None, path: str = None, verbose: bool = True):
        '''Loads self from either user specified cmd line args (training) or a text file (pre-trained)'''
        if cmd_args:
            iter_with_vars = vars(cmd_args).items()
        elif path:
            iter_with_vars = []
            with open(path) as fread:
                for line in fread:
                    key, val = line.strip('\n').split(': ')
                    val = self.guess_type(val)
                    iter_with_vars.append((key, val))
        else:
            assert cmd_args is None and path is None
            raise Exception('Both arguments are None')

        for var_name, var_value in iter_with_vars:
            self.assign_variable(var_name, var_value)
            if verbose:
                print(f'    {var_name}: {var_value}')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(cfg, datasets, dataloaders, device, save_model_path):
    model = RNN(
        cfg.model_type, cfg.input_dim, cfg.hidden_dim, cfg.n_layers, cfg.drop_p, cfg.output_dim, cfg.bi_dir
    )
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_metric = 0.0
    best_epoch = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(cfg.num_epochs):
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            # running_corrects = 0
            y_pred = []
            y_true = []

            # Iterate over data.
            for batch in dataloaders[phase]:
                inputs = batch['inputs'].to(device)
                targets = batch['targets'][cfg.task].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs, hiddens = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, targets)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                # running_corrects += torch.sum(preds == targets.data)
                y_pred.extend(preds.tolist())
                y_true.extend(targets.tolist())

            # if phase == 'train':
            #     scheduler.step()

            # epoch_acc = running_corrects.double() / len(datasets[phase])
            epoch_loss = running_loss / len(datasets[phase])
            f1_ep = f1_score(y_true, y_pred, average='weighted')
            precision_ep = precision_score(y_true, y_pred, average='weighted')
            recall_ep = recall_score(y_true, y_pred, average='weighted')
            accuracy_ep = accuracy_score(y_true, y_pred)

            # print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            print(f'({phase} @ {epoch+1}): L: {epoch_loss:3f}; A: {accuracy_ep:3f}; R: {recall_ep:3f}; ' +
                  f'P: {precision_ep:3f}; F1: {f1_ep:3f}')

            # deep copy the model
            if phase == 'valid' and f1_ep > best_metric:
                best_metric = f1_ep
                best_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())

    print(f'Best val Metric {best_metric:3f} @ {best_epoch+1}\n')

    # load best model weights and saves it
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), save_model_path)
    print(f'model is saved @ {save_model_path}')
    return best_metric


def predict(cfg, model_path, loader, device, save_path):
    print(f'Saving predictions @ {save_path}')
    # define model
    model = RNN(
        cfg.model_type, cfg.input_dim, cfg.hidden_dim, cfg.n_layers, cfg.drop_p, cfg.output_dim, cfg.bi_dir
    )
    model = model.to(device)
    # load the model
    model.load_state_dict(torch.load(model_path))

    # just to be sure
    model.eval()

    predictions = {
        'Object': [],
        'Sequence': [],
    }

    for c in range(cfg.output_dim):
        predictions[f'{cfg.task}_prob_{c}'] = []

    for batch in loader:
        inputs = batch['inputs'].to(device)

        with torch.set_grad_enabled(False):
            outputs, hiddens = model(inputs)
            _, preds = torch.max(outputs, 1)
            softmaxed = torch.nn.functional.softmax(outputs, dim=-1)

        for i in range(len(batch['paths'])):
            sequence = pathlib.Path(batch['paths'][i]).stem.replace('_audio', '')
            predictions['Object'].append(batch['containers'][i])
            predictions['Sequence'].append(sequence.replace('_vggish', ''))
            for c in range(cfg.output_dim):
                predictions[f'{cfg.task}_prob_{c}'].append(softmaxed[i, c].item())

    predictions_dataset = pd.DataFrame.from_dict(predictions).sort_values(['Object', 'Sequence'])
    predictions_dataset.to_csv(save_path, index=False)
    # returning the dataset because it will be useful for test-time prediction averaging
    return predictions_dataset


def experiment(cfg, fold, use_pretrained, predict_on_private):
    print(fold)
    set_seed(cfg.seed)

    device = torch.device(cfg.device)

    datasets = {
        'train': AudioDataset(cfg.data_root, fold['train'], 'train'),
        'valid': AudioDataset(cfg.data_root, fold['valid'], 'valid'),
        'public_test': AudioDataset(cfg.data_root, fold['public_test'], 'public_test'),
    }

    dataloaders = {
        'train': DataLoader(
            datasets['train'], batch_size=cfg.batch_size, shuffle=True,
            collate_fn=datasets['train'].collate_fn
        ),
        'valid': DataLoader(
            datasets['valid'], batch_size=cfg.batch_size, shuffle=False,
            collate_fn=datasets['valid'].collate_fn
        ),
        'public_test': DataLoader(
            datasets['public_test'], batch_size=cfg.batch_size, shuffle=False,
            collate_fn=datasets['public_test'].collate_fn
        )
    }

    if predict_on_private:
        datasets['private_test'] = AudioDataset(cfg.data_root, fold['private_test'], 'private_test')
        dataloaders['private_test'] = DataLoader(
            datasets['private_test'], batch_size=cfg.batch_size, shuffle=False,
            collate_fn=datasets['private_test'].collate_fn
        )

    model_path = f'./predictions/{cfg.init_time}/{cfg.task}_{"_".join([str(i) for i in fold["train"]])}_pretrained_model.pt'

    if use_pretrained:
        print(f'Using pre-trained model from {model_path}')
        best_metric = -1.0
    else:
        best_metric = train(cfg, datasets, dataloaders, device, model_path)

    # make predictions
    test_predictions = {}
    predict(
        cfg, model_path, dataloaders['train'], device,
        f'./predictions/{cfg.init_time}/{cfg.task}_train_{"_".join([str(i) for i in fold["train"]])}_vggish.csv')
    predict(
        cfg, model_path, dataloaders['valid'], device,
        f'./predictions/{cfg.init_time}/{cfg.task}_valid_{"_".join([str(i) for i in fold["valid"]])}_vggish.csv')
    test_predictions['public_test'] = predict(
        cfg, model_path, dataloaders['public_test'], device,
        f'./predictions/{cfg.init_time}/{cfg.task}_public_test_trained_on_{"_".join([str(i) for i in fold["train"]])}_vggish.csv')

    if predict_on_private:
        test_predictions['private_test'] = predict(
            cfg, model_path, dataloaders['private_test'], device,
            f'./predictions/{cfg.init_time}/{cfg.task}_private_test_trained_on_{"_".join([str(i) for i in fold["train"]])}_vggish.csv')

    return best_metric, test_predictions


def run_kfold(cfg, use_pretrained=False, predict_on_private=None):
    save_results = os.path.join('./predictions/', str(cfg.init_time))
    if use_pretrained is False:
        os.makedirs(save_results)
    cfg.save_self_to(os.path.join(save_results, 'cfg.txt'))
    folds = [
        {'train': [1, 2, 4, 5, 7, 8], 'valid': [3, 6, 9], 'public_test': [10, 11, 12], 'private_test': [13, 14, 15]},
        {'train': [1, 3, 4, 6, 7, 9], 'valid': [2, 5, 8], 'public_test': [10, 11, 12], 'private_test': [13, 14, 15]},
        {'train': [2, 3, 5, 6, 8, 9], 'valid': [1, 4, 7], 'public_test': [10, 11, 12], 'private_test': [13, 14, 15]}
    ]
    best_fold_metrics_and_test_predictions = [experiment(cfg, fold, use_pretrained, predict_on_private) for fold in folds]
    best_fold_metrics = [metric for metric, _ in best_fold_metrics_and_test_predictions]
    test_predictions = [preds for _, preds in best_fold_metrics_and_test_predictions]
    if use_pretrained is False:
        print(f'Average of Best Metrics on Each Valid Set: {np.mean(best_fold_metrics):4f}, {cfg.init_time}')

    def _agg_preds_from_all_folds(phase='public_test'):
        # averaging predictions from all tree folds
        aggregated_dataset = test_predictions[0][phase].copy()

        for c in range(cfg.output_dim):
            column = f'{cfg.task}_prob_{c}'
            fold_columns = [test_predictions[f][phase][column] for f in range(len(folds))]
            aggregated_dataset[column] = np.mean(fold_columns, axis=0)

        aggregated_dataset.to_csv(
            os.path.join(save_results, f'{cfg.task}_{phase}_agg_vggish.csv'),
            index=False
        )
        print(f'Saved test results @ {os.path.join(save_results, f"{cfg.task}_{phase}_agg_vggish.csv")}')

    _agg_preds_from_all_folds('public_test')

    if predict_on_private:
        _agg_preds_from_all_folds('private_test')


if __name__ == "__main__":
    # Reproduce the best experiment
    # if True, will use the pre-trained model and make predictions, if False, will train the model
    use_pretrained = True
    exp_name = 200903162117
    cfg = Config()
    cfg.load_from(path=f'./predictions/{exp_name}/cfg.txt')
    # replacing the time with the old_time + current_time such that there is no collision
    if use_pretrained:
        cfg.init_time = exp_name
    else:
        cfg.init_time = f'{cfg.init_time}_{strftime("%y%m%d%H%M%S", localtime())}'
    # Expected average of Best Metrics on Each Valid Set: 0.755171 @ 200903162117
    run_kfold(cfg, use_pretrained, get_cmd_args().predict_on_private)

    # # Experiment with other parameters
    # cfg = Config()
    # cfg.assign_variable('task', 'flvl')
    # cfg.assign_variable('output_dim', 3)
    # cfg.assign_variable('model_type', 'GRU')
    # cfg.assign_variable('bi_dir', False)
    # cfg.assign_variable('device', 'cuda:0')
    # cfg.assign_variable('data_root', './vggish_features')
    # cfg.assign_variable('batch_size', 64)
    # cfg.assign_variable('input_dim', 128)
    # cfg.assign_variable('hidden_dim', 512)
    # cfg.assign_variable('n_layers', 5)
    # cfg.assign_variable('drop_p', 0.0) # results will be irreproducible
    # cfg.assign_variable('num_epochs', 25)
    # cfg.assign_variable('seed', 1337)
    # run_kfold(cfg)
