from nanigonet.language_info import LanguageInfo

import os
import random
import sys

from pathlib import Path

TRAIN_DIR = Path('data/train')


def get_num_lines_from_rank(rank):
    if not rank:
        return 2000

    rank = int(rank)
    if rank <= 10:
        return 6000
    elif 10 < rank <= 20:
        return 3000


def get_deduped_lines(file_path, num_lines=1000):
    lines = set()
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if len(line) > 1024:
                continue
            lines.add(line)

    lines = list(lines)
    random.shuffle(lines)
    lines = lines[:num_lines]

    return lines


def main():
    random.seed(1)

    for info in LanguageInfo.values():
        num_lines = get_num_lines_from_rank(info['rank'])
        print(f"Creating training data for {info['id']}. num_lines={num_lines}", file=sys.stderr)

        if info['type'] == 'h':
            target_dir = TRAIN_DIR / info['id']
        else:
            target_dir = TRAIN_DIR / f"p-{info['id']}"

        if not os.path.exists(target_dir):
            print(f"Directory for {info['id']} does not exist. Skipping.", file=sys.stderr)
            continue

        all_lines = []

        tatoeba_path = target_dir / 'tatoeba.txt'
        if os.path.exists(tatoeba_path):
            new_lines = get_deduped_lines(tatoeba_path, num_lines=num_lines)
            all_lines.extend(new_lines)

        w2c_path = target_dir / 'w2c.txt'
        if os.path.exists(w2c_path):
            new_lines = get_deduped_lines(w2c_path, num_lines=num_lines)
            all_lines.extend(new_lines)

        github_path = target_dir / 'github.small.txt'
        if os.path.exists(github_path):
            new_lines = get_deduped_lines(github_path, num_lines=num_lines)
            all_lines.extend(new_lines)

        with open(target_dir / 'combined.txt', mode='w') as f:
            for line in all_lines:
                f.write(line)
                f.write('\n')


if __name__ == '__main__':
    main()
