import json
from typing import Dict, Iterable, List

from allennlp.common.file_utils import cached_path
from allennlp.data import DatasetReader, Instance, TokenIndexer
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, CharacterTokenizer
from allennlp.data.fields import TextField, SequenceLabelField


@DatasetReader.register('nanigonet')
class NanigoNetDatasetReader(DatasetReader):

    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_token_len: int = 512):
        super().__init__(lazy)

        self._tokenizer = tokenizer or CharacterTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._max_token_len = max_token_len

    def text_to_instance(self, tokens: List[Token], tags: List[str]=None) -> Instance:

        if len(tokens) > self._max_token_len:
            tokens = tokens[:self._max_token_len]
            print(f'Length of tokens exceeded the limit {self._max_token_len}. Truncating...')
            if tags:
                tags = tags[:self._max_token_len]

        fields = {}

        text_field = TextField(tokens, self._token_indexers)
        fields['tokens'] = text_field
        if tags:
            fields['tags'] = SequenceLabelField(tags, text_field)

        return Instance(fields)

    def _read(self, file_path: str) -> Iterable[Instance]:
        file_path = cached_path(file_path)

        with open(file_path) as f:
            for line in f:
                data = json.loads(line)
                tokens = self._tokenizer.tokenize(data['text'])
                tags = data.get('labels')

                yield self.text_to_instance(tokens, tags)


if __name__ == '__main__':
    # test dataset reader
    import sys

    reader = NanigoNetDatasetReader()
    for inst in reader.read(sys.argv[1]):
        print(inst)
