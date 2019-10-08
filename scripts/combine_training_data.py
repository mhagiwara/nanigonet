import json
import os
import random
import sys
from pathlib import Path

from nanigonet.language_info import LanguageInfo

TRAIN_DIR = Path('data/train')


def main():
    random.seed(0)

    # First, read all the combined files per language ...
    lines_and_langs = []
    for info in LanguageInfo.values():
        print(f"Combining training data for {info['id']} ...", file=sys.stderr)

        if not os.path.exists(TRAIN_DIR / info['id']):
            print(f"Directory for {info['id']} does not exist. Skipping.", file=sys.stderr)
            continue

        combined_path = TRAIN_DIR / info['id'] / 'combined.txt'
        if os.path.exists(combined_path):
            with open(combined_path) as f:
                for line in f:
                    text = line.strip()
                    text = text + ' '  # add a whitespace to account for punctuation.
                    lines_and_langs.append((text, info['id']))

    # Second, shuffle them
    random.shuffle(lines_and_langs)

    # Third, split them into chunks
    last_split_i = 0
    last_split_j = 0
    dataset = []
    for i, (line, lang) in enumerate(lines_and_langs):
        for j in range(len(line) + 1):
            # split at position j means to split the text BEFORE text[j]
            if random.random() >= 0.008:
                continue

            # make a split at [i, j]
            if last_split_i == i:
                text = line[last_split_j:j]
                labels = [lang] * (j - last_split_j)
            else:
                text = ''
                labels = []
                for k in range(last_split_i, i+1):
                    if k == last_split_i:
                        partial_line, partial_lang = lines_and_langs[last_split_i]
                        partial_line = partial_line[last_split_j:]
                    elif k == i:
                        partial_line, partial_lang = line[:j], lang
                    else:
                        partial_line, partial_lang = lines_and_langs[k]

                    text += partial_line
                    labels.extend([partial_lang] * len(partial_line))

            assert len(text) == len(labels)
            dataset.append((text, labels))

            last_split_i = i
            last_split_j = j

    # Finally, print out the results

    for text, labels in dataset:
        data = {
            'text': text,
            'labels': labels
        }
        print(json.dumps(data, ensure_ascii=False))


if __name__ == '__main__':
    main()
