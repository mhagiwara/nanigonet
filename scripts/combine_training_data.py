from nanigonet.language_info import LanguageInfo

import json
import sys
from pathlib import Path
import os

TRAIN_DIR = Path('data/train')


def main():
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

                    data = {
                        'text': text,
                        'labels': [
                            {'startOffset': 0,
                             'endOffset': len(line),
                             'langId': info['id']}
                        ]
                    }
                    print(json.dumps(data, ensure_ascii=False))


if __name__ == '__main__':
    main()
