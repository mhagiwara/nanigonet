import argparse
import json
import sys
from nanigonet import NanigoNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('archive_file', type=str,
                        help='the archived model to make predictions with')
    parser.add_argument('--top-k', type=int, default=3,
                        help='the number of best solutions to return')
    parser.add_argument('--cuda-device', type=int, default=-1,
                        help='id of GPU to use (if any)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='The batch size to use for processing')
    args = parser.parse_args()

    net = NanigoNet(model_path=args.archive_file,
                    top_k=args.top_k,
                    cuda_device=args.cuda_device)

    batch = []
    for line in sys.stdin:
        text = line[:-1]
        batch.append(text)
        if len(batch) == args.batch_size:
            results = net.predict_batch(batch)
            for result in results:
                print(json.dumps(result))
            batch = []

    if batch:
        results = net.predict_batch(batch)
        for result in results:
            print(json.dumps(result))


if __name__ == '__main__':
    main()
