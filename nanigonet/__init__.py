from nanigonet.dataset_reader import NanigoNetDatasetReader
import numpy as np

from collections import Counter

import numpy as np
from allennlp.common.util import prepare_environment
from allennlp.data import Instance
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import CharacterTokenizer
from allennlp.models import load_archive


def softmax(x, axis=None):
    e_x = np.exp(x - np.max(x, axis=axis).reshape((-1, 1)))
    return e_x / e_x.sum(axis=axis).reshape((-1, 1))


class NanigoNet:
    def __init__(self, model_path, top_k=3):
        archive = load_archive(model_path)

        config = archive.config
        prepare_environment(config)
        model = archive.model
        model.eval()

        self.model = model

        self._tokenizer = CharacterTokenizer()
        self._token_indexers = {'tokens': SingleIdTokenIndexer()}
        self._id_to_label = model.vocab.get_index_to_token_vocabulary(namespace='labels')
        self._top_k = top_k

    def predict(self, text):
        tokens = self._tokenizer.tokenize(text)
        instance = Instance({'tokens': TextField(tokens, self._token_indexers)})

        result = self.model.forward_on_instance(instance)

        char_probs = []
        probs = softmax(result['logits'], axis=-1)
        for probs_per_char in probs:
            counter = Counter({self._id_to_label[label_id]: float(prob)
                               for label_id, prob in enumerate(probs_per_char)})
            char_probs.append(dict(counter.most_common(self._top_k)))

        total_probs = probs.mean(axis=0)
        counter = Counter({self._id_to_label[label_id]: float(prob)
                           for label_id, prob in enumerate(total_probs)})

        return {
            'char_probs': char_probs,
            'probs': dict(counter.most_common(3)),
            'tags': result['tags'],
            'prediction': counter.most_common(1)[0][0]
        }
