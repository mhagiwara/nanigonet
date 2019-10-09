import langdetect
from pathlib import Path

TRAIN_DIR = Path('data/train')


def main():
    parent_dir = TRAIN_DIR / 'cmn'

    for target_file in ['tatoeba.txt', 'w2c.txt']:

        hant_file = open(TRAIN_DIR / 'cmn-hant' / target_file, mode='w')
        hans_file = open(TRAIN_DIR / 'cmn-hans' / target_file, mode='w')

        with open(parent_dir / target_file) as f:
            for line in f:
                text = line.rstrip()
                if not text:
                    continue

                try:
                    lang = langdetect.detect(text)
                except langdetect.lang_detect_exception.LangDetectException:
                    continue

                if lang in {'zh-tw', 'ko'}:
                    hant_file.write(text)
                    hant_file.write('\n')
                elif lang == 'zh-cn':
                    hans_file.write(text)
                    hans_file.write('\n')

        hant_file.close()
        hans_file.close()


if __name__ == '__main__':
    main()
