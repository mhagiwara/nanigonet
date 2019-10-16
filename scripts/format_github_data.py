import json
import os
import sys
from collections import Counter
from pathlib import Path

TRAIN_DIR = Path('data/train')


def count_popular_languages():
    language_counts = Counter()

    for line in sys.stdin:
        data = json.loads(line)
        for _, path_after in data['paths']:
            extension = path_after.rsplit('.', 1)[-1]
            language_counts[extension] += 1

    for language, counts in language_counts.most_common():
        print(language, counts)


def main():
    extension_mapping = {
        'c': 'c',
        'h': 'c',
        'cc': 'cpp',
        'cpp': 'cpp',
        'cs': 'cs',
        'css': 'css',
        'go': 'go',
        'hs': 'hs',
        'html': 'html',
        'java': 'java',
        'js': 'js',
        'm': 'm',
        'php': 'php',
        'py': 'py',
        'rb': 'rb',
        'rs': 'rs',
        'scala': 'scala',
        'sh': 'sh',
        'swift': 'swift',
        'ts': 'ts',
        'xml': 'xml',
    }

    lang_id_to_file = {}

    for line in sys.stdin:
        data = json.loads(line)
        for (_, diff_after), (_, path_after) in zip(data['diffs'], data['paths']):
            if len(diff_after.strip()) < 5 or len(diff_after) > 256:
                continue

            extension = path_after.rsplit('.', 1)[-1]
            lang_id = extension_mapping.get(extension)
            if lang_id is None:
                continue

            # Create 'data/train/p-{lang_id}' directory if not exists
            if not os.path.exists(TRAIN_DIR / f'p-{lang_id}'):
                os.makedirs(TRAIN_DIR / f'p-{lang_id}')

            if lang_id in lang_id_to_file:
                f = lang_id_to_file[lang_id]
            else:
                f = open(TRAIN_DIR / f'p-{lang_id}' / 'github.txt', mode='w')
                lang_id_to_file[lang_id] = f

            f.write(diff_after)
            f.write('\n')

    for f in lang_id_to_file.values():
        f.close()


if __name__ == '__main__':
    main()
