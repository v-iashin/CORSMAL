from glob import glob
import os
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


class RGBDataset(torch.utils.data.Dataset):

    def __init__(self, data_root, types, phase):
        super(RGBDataset, self).__init__()
        self.data_root = data_root
        self.types = types
        self.phase = phase
        self.dataset = self.make_meta_dataset()

    def __getitem__(self, index):
        meta_i = self.dataset.iloc[index].to_dict()
        # contruct paths to cameras from the same event
        base_path = os.path.join(self.data_root, meta_i['path']).replace('_c1_rgb.npy', '')
        paths = [
            f'{base_path}_c1_rgb.npy',
            f'{base_path}_c2_rgb.npy',
            f'{base_path}_c3_rgb.npy',
            f'{base_path}_c4_rgb.npy'
        ]
        # (cam_num, T, D)
        inputs = torch.cat([torch.from_numpy(np.load(path))[np.newaxis, ...] for path in paths])
        # for pad_sequence we will flip dims
        # (T, cam_num, D)
        inputs = inputs.permute((1, 0, 2))

        if self.phase == 'test':
            targets = None
        else:
            targets = {
                'ftype': int(meta_i['ftype']),
                'flvl': int(meta_i['flvl']),
            }

        item = {
            'targets': targets,
            'inputs': inputs,
            'path': base_path,
            'container': meta_i['container']
        }

        return item

    def __len__(self):
        return len(self.dataset)

    def collate_fn(self, batch: list) -> dict:
        out = {
            'inputs': None,
            'targets': {},
            'paths': [item['path'] for item in batch],
            'containers': [item['container'] for item in batch]
        }

        padded_seq = pad_sequence([item['inputs'] for item in batch], batch_first=True, padding_value=-10)
        # out['inputs'] = pack_padded_sequence(padded_seq, lengths, batch_first=True, enforce_sorted=False)
        # (B, cam_num, T, D) <- (B, T, cam_num, D)
        out['inputs'] = padded_seq.permute((0, 2, 1, 3))

        if self.phase == 'test':
            out['targets'] = None
        else:
            out['targets']['ftype'] = torch.tensor([item['targets']['ftype'] for item in batch])
            out['targets']['flvl'] = torch.tensor([item['targets']['flvl'] for item in batch])

        return out

    def make_meta_dataset(self):
        dataset = {
            'path': [],
            'container': [],
            'subject': [],
            'ftype': [],
            'flvl': [],
            'background': [],
            'lights': [],
        }
        for container_type in self.types:
            # form list of paths (will use only paths to c1 but later use all 4 cameras)
            paths = sorted(glob(os.path.join(self.data_root, f'{container_type}', 'r21d_rgb', '*c1_rgb.npy')))
            paths = [path.replace(f'{self.data_root}/', '') for path in paths]
            # parse the file names
            for path in paths:
                # '/X/vggish || sX_fiX_fuX_bX_lX_vggish.npy'
                _, filename = os.path.split(path)
                # s2 fi1 fu1 b1 l0 <- 'sX_fiX_fuX_bX_lX.mp4'
                # 0000 cX <- 0000_c1.mp4
                if 1 <= container_type <= 9:
                    subj, ftype, flevel, backgr, lights = filename.replace('_c1_rgb.npy', '').split('_')
                    dataset['path'].append(path)
                    dataset['container'].append(container_type)
                    dataset['subject'].append(subj.replace('s', ''))
                    dataset['ftype'].append(ftype.replace('fi', ''))
                    dataset['flvl'].append(flevel.replace('fu', ''))
                    dataset['background'].append(backgr.replace('b', ''))
                    dataset['lights'].append(lights.replace('l', ''))
                else:
                    dataset['path'].append(path)
                    dataset['container'].append(container_type)
                    dataset['subject'].append(None)
                    dataset['ftype'].append(None)
                    dataset['flvl'].append(None)
                    dataset['background'].append(None)
                    dataset['lights'].append(None)

        dataset = pd.DataFrame.from_dict(dataset)
        return dataset


if __name__ == "__main__":
    DATA_ROOT = './r21d_rgb_features'

    train = RGBDataset(DATA_ROOT, [1, 2, 3, 4, 5, 6], 'train')
    valid = RGBDataset(DATA_ROOT, [7, 8, 9], 'valid')
    test = RGBDataset(DATA_ROOT, [10, 11, 12], 'test')

    train_loader = DataLoader(train, batch_size=4, shuffle=True, collate_fn=train.collate_fn)
    valid_loader = DataLoader(valid, batch_size=2, shuffle=False, collate_fn=valid.collate_fn)
    test_loader = DataLoader(test, batch_size=2, shuffle=False, collate_fn=test.collate_fn)

    # tests
    idx = 50
    print(train[idx])
    print(train[idx]['inputs'].shape)
    print(valid[idx])
    print(valid[idx]['inputs'].shape)
    print(test[idx])
    print(test[idx]['inputs'].shape)

    loader = train_loader
    loader = test_loader

    for batch in loader:
        print(batch)
        break
