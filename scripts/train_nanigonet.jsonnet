{
    "dataset_reader": {
        "type": "nanigonet"
    },
    "train_data_path": "data/train.jsonl",
    "validation_data_path": "data/dev.jsonl",

    "vocabulary" : {
        "min_count": {"tokens": 3}
    },

    "model": {
        "type": "crf_tagger",
        "constrain_crf_decoding": false,
        "calculate_span_f1": false,
        "dropout": 0.5,
        "include_start_end_transitions": false,

        "text_field_embedder": {
            "tokens": {
                "type": "embedding",
                "embedding_dim": 256
            }
        },

        "encoder": {
            "type": "lstm",
            "input_size": 256,
            "hidden_size": 256,
            "num_layers": 2,
            "bidirectional": true,
            "dropout": 0.5,
        }
    },
    "iterator": {
        "type": "bucket",
        "batch_size": 96,
        "sorting_keys": [["tokens", "num_tokens"]]
    },

    "trainer": {
        "optimizer": {
            "type": "adam",
            "lr": 0.0005
        },
        "validation_metric": "+accuracy",
        "num_epochs": 100,
        "patience": 10,
        "cuda_device": 0
    }
}
