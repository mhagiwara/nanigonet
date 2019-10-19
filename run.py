import json
import sys
from nanigonet import NanigoNet


def main():
    net = NanigoNet(model_path=sys.argv[1])

    for line in sys.stdin:
        text = line[:-1]

        data = net.predict(text)
        print(json.dumps(data))


if __name__ == '__main__':
    main()
