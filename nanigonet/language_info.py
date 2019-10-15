import csv

LanguageInfo = {}


with open('./languages.tsv') as f:
    reader = csv.reader(f, delimiter='\t')
    header = next(reader)
    for row in reader:
        data = dict(zip(header, row))
        LanguageInfo[data['id']] = data


if __name__ == '__main__':
    for lang_id, info in LanguageInfo.items():
        print(lang_id, info)
