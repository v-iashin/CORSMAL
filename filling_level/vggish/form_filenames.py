import argparse
from glob import glob
import os

def make_file_with_paths(path_pattern, save_path):
    file_paths = glob(path_pattern)
    with open(save_path, 'w') as outf:
        for path in file_paths:
            outf.write(f'{path}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Extract features')
    parser.add_argument('--data_root', help='Path to CORSMAL dataset (the folder should contain: 1/ 2/ ...')
    args = parser.parse_args()

    os.makedirs('filepaths', exist_ok=True)

    for cont_id in range(1, 12+1):
        # it will create ./filepaths dir (already present in the repo) with file paths to each container
        make_file_with_paths(
            os.path.join(args.data_root, cont_id, 'rgb/*.mp4'), f'./filepaths/{cont_id}_rgb_file_paths.txt'
        )
