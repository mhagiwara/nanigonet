import csv
import os
import sys
import zlib
from pathlib import Path

import requests

from nanigonet.language_info import LanguageInfo

TRAIN_DIR = Path('data/train')


def download_for_language(lang_id, w2c_id):
    """
    Download W2C Wiki corpus and deflate it.
    Note that the language ID for W2C corpus might differ from our ISO-639-3 ID.
    """
    print(f'Downloading W2C corpus for {lang_id}...', file=sys.stderr)

    url_prefix = 'http://ufal.mff.cuni.cz/~majlis/w2c/download.php'
    response = requests.get(f"{url_prefix}?lang={w2c_id}&type=wiki")

    try:
        with open(TRAIN_DIR / lang_id / f'w2c.txt', mode='wb') as f:
            f.write(zlib.decompress(response.content, zlib.MAX_WBITS | 32))
    except zlib.error as e:
        print(e, file=sys.stderr)


def main():
    for info in LanguageInfo.values():
        if info['w2c']:
            # Create 'data/train/{lang_id}' directory if not exists
            if not os.path.exists(TRAIN_DIR / info['id']):
                os.makedirs(TRAIN_DIR / info['id'])

            download_for_language(info['id'], info['w2c'])


if __name__ == '__main__':
    main()
