from nanigonet.language_info import LanguageInfo

import os
import random
import sys

from pathlib import Path

TRAIN_DIR = Path('data/train')


def get_deduped_lines(file_path):
    lines = set()
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line = line[:256]
            lines.add(line)

    lines = list(lines)
    random.shuffle(lines)
    lines = lines[:10000]

    return lines


def main():
    random.seed(1)

    for info in LanguageInfo.values():
        print(f"Creating training data for {info['id']} ...", file=sys.stderr)

        if not os.path.exists(TRAIN_DIR / info['id']):
            print(f"Directory for {info['id']} does not exist. Skipping.", file=sys.stderr)
            continue

        all_lines = []

        tatoeba_path = TRAIN_DIR / info['id'] / 'tatoeba.txt'
        if os.path.exists(tatoeba_path):
            new_lines = get_deduped_lines(tatoeba_path)
            all_lines.extend(new_lines)

        w2c_path = TRAIN_DIR / info['id'] / 'w2c.txt'
        if os.path.exists(w2c_path):
            new_lines = get_deduped_lines(w2c_path)
            all_lines.extend(new_lines)

        with open(TRAIN_DIR / info['id'] / 'combined.txt', mode='w') as f:
            for line in all_lines:
                f.write(line)
                f.write('\n')


if __name__ == '__main__':
    main()
