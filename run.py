import sys

from allennlp.common.util import prepare_environment
from allennlp.data import Instance
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import CharacterTokenizer
from allennlp.models import load_archive


def main():
    archive_path = sys.argv[1]

    # Load the archive & model
    archive = load_archive(archive_path)

    config = archive.config
    prepare_environment(config)
    model = archive.model
    model.eval()

    tokenizer = CharacterTokenizer()
    token_indexers = {'tokens': SingleIdTokenIndexer()}

    for line in sys.stdin:
        text = line[:-1]
        tokens = tokenizer.tokenize(text)
        instance = Instance({'tokens': TextField(tokens, token_indexers)})

        result = model.forward_on_instance(instance)
        probs = result['logits']
        print(result)


if __name__ == '__main__':
    main()
