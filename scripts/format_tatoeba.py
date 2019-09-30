import os
from pathlib import Path

from nanigonet.language_info import LanguageInfo

TRAIN_DIR = Path('data/train')


def main():
    tatoeba_to_nanigonet_id = {}
    for info in LanguageInfo.values():
        if info['tatoeba']:
            tatoeba_to_nanigonet_id[info['tatoeba']] = info['id']

    nanigonet_id_to_file = {}

    with open('data/sentences.csv') as f:
        for line in f:
            _, tatoeba_id, text = line.rstrip().split('\t', maxsplit=2)

            if tatoeba_id not in tatoeba_to_nanigonet_id:
                continue

            nanigonet_id = tatoeba_to_nanigonet_id[tatoeba_id]
            if nanigonet_id not in nanigonet_id_to_file:
                if not os.path.exists(TRAIN_DIR / nanigonet_id):
                    os.makedirs(TRAIN_DIR / nanigonet_id)

                f = open(TRAIN_DIR / nanigonet_id / 'tatoeba.txt', mode='w')
                nanigonet_id_to_file[nanigonet_id] = f

            f = nanigonet_id_to_file[nanigonet_id]
            f.write(text)
            f.write('\n')

    for f in nanigonet_id_to_file.values():
        f.close()


if __name__ == '__main__':
    main()
