import argparse
import copy
import os
import pathlib
import random
import time
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
    parser.add_argument('--train_types', default=[1, 2, 4, 5], type=int, nargs='+')
    parser.add_argument('--valid_types', default=[3, 6, 9], type=int, nargs='+')
    parser.add_argument('--test_types', default=[10, 11, 12], type=int, nargs='+')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--data_root', default='/home/nvme/vladimir/corsmal/features')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--input_dim', default=128, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--n_layers', default=5, type=int)
    parser.add_argument('--drop_p', default=0.0, type=float)
    parser.add_argument('--num_epochs', default=50, type=int)
    parser.add_argument('--seed', default=1337, type=int)
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

    def load_from(self, cmd_args: argparse.Namespace = None, path: str = None, verbose: bool = True):
        '''Loads self from either user specified cmd line args (training) or a text file (pre-trained)'''
        if cmd_args:
            iter_with_vars = vars(cmd_args).items()
        elif path:
            iter_with_vars = []
            with open(path) as fread:
                for line in fread:
                    key, val = line.strip('\n').split(': ')
                    # step_size: 15. '15' will be interpreted as string -- preventing this
                    # TODO: what to do if float 1e-4
                    if all(map(str.isdigit, val)):
                        val = int(val)
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


def experiment(cfg):
    set_seed(cfg.seed)

    device = torch.device(cfg.device)

    datasets = {
        'train': AudioDataset(cfg.data_root, cfg.train_types, 'train'),
        'valid': AudioDataset(cfg.data_root, cfg.valid_types, 'valid'),
        'test': AudioDataset(cfg.data_root, cfg.test_types, 'test'),
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
        'test': DataLoader(
            datasets['test'], batch_size=cfg.batch_size, shuffle=False,
            collate_fn=datasets['test'].collate_fn
        )
    }

    model = RNN(
        cfg.model_type, cfg.input_dim, cfg.hidden_dim, cfg.n_layers, cfg.drop_p, cfg.output_dim, cfg.bi_dir
    )
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_acc = 0.0
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(cfg.num_epochs):
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

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
                running_corrects += torch.sum(preds == targets.data)
            # if phase == 'train':
            #     scheduler.step()

            epoch_loss = running_loss / len(datasets[phase])
            epoch_acc = running_corrects.double() / len(datasets[phase])

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    # phase
    phase = 'test'

    # ## TESTING
    # Iterate over data.
    # Object,Sequence,Container capacity [mL],Container mass [g],Filling type,Filling level [%],Filling mass [g]
    test_prediction = {
        'Object': [],
        'Sequence': [],
    }

    for c in range(cfg.output_dim):
        test_prediction[f'{cfg.task}_prob_{c}'] = []

    for batch in dataloaders[phase]:
        inputs = batch['inputs'].to(device)

        with torch.set_grad_enabled(False):
            outputs, hiddens = model(inputs)
            _, preds = torch.max(outputs, 1)
            softmaxed = torch.nn.functional.softmax(outputs, dim=-1)

        for i in range(len(batch['paths'])):
            sequence = pathlib.Path(batch['paths'][i]).stem.replace('_audio', '')
            test_prediction['Object'].append(batch['containers'][i])
            test_prediction['Sequence'].append(sequence.replace('_vggish', ''))
            for c in range(cfg.output_dim):
                test_prediction[f'{cfg.task}_prob_{c}'].append(softmaxed[i, c].item())

    pd.DataFrame.from_dict(test_prediction).to_csv(f'{cfg.task}_vggish.csv', index=False)


if __name__ == "__main__":
    cfg = Config()
    cfg.load_from(cmd_args=get_cmd_args())
    experiment(cfg)
