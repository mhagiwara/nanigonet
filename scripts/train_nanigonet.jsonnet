{
    "dataset_reader": {
        "type": "nanigonet",
        "max_token_len": 512
    },
    "train_data_path": "data/train-large.jsonl",
    "validation_data_path": "data/dev-large.jsonl",

    "vocabulary" : {
        "min_count": {"tokens": 3}
    },

    "model": {
        "type": "simple_tagger",

        "text_field_embedder": {
            "tokens": {
                "type": "embedding",
                "embedding_dim": 256
            }
        },

        "encoder": {
            "type": "gated-cnn-encoder",
            "input_dim": 256,
            "dropout": 0.5,
            "layers": [ [[5, 256]], [[5, 256], [5, 256]], [[5, 256], [5, 256]], [[5, 256], [5, 256]], [[5, 256], [5, 256]],
                                    [[5, 256], [5, 256]], [[5, 256], [5, 256]], [[5, 256], [5, 256]], [[5, 256], [5, 256]],
                                    [[5, 256], [5, 256]], [[5, 256], [5, 256]] ]
        }
    },
    "iterator": {
        "type": "bucket",
        "batch_size": 128,
        "padding_noise": 0.1,
        "sorting_keys": [["tokens", "num_tokens"]]
    },

    "trainer": {
        "optimizer": {
            "type": "adam",
            "lr": 0.0001
        },
        "validation_metric": "+accuracy",
        "grad_clipping": 0.1,
        "num_epochs": 100,
        "patience": 10,
        "cuda_device": 0
    }
}
