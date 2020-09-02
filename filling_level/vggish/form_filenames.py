from glob import glob
import os

def make_file_with_paths(path_pattern, save_path):
    file_paths = glob(path_pattern)
    with open(save_path, 'w') as outf:
        for path in file_paths:
            outf.write(f'{path}\n')


if __name__ == "__main__":
    os.makedirs('filepaths', exist_ok=True)
    for i in range(1, 12+1):
        make_file_with_paths(
            f'/home/nvme/vladimir/corsmal/{i}/audio/*.wav', f'./filepaths/{i}_audio_file_paths.txt'
        )
        make_file_with_paths(
            f'/home/nvme/vladimir/corsmal/{i}/rgb/*.mp4', f'./filepaths/{i}_rgb_file_paths.txt'
        )
