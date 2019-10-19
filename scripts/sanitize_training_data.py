import sys
from nanigonet import NanigoNet


def main():
    net = NanigoNet(model_path=sys.argv[1])
    expected = sys.argv[2]

    for line in sys.stdin:
        text = line[:-1]

        if not text:
            print(text)
            continue

        data = net.predict(text)
        prediction = data['prediction']
        if prediction != expected:
            print(f'!!!<{prediction}>{text}')
        else:
            print(text)


if __name__ == '__main__':
    main()
