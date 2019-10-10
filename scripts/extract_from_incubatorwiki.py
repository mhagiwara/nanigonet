import sys


def main():
    target_prefix = sys.argv[1]
    inside_target = False
    for line in sys.stdin:
        line = line.rstrip()
        if line.startswith('<doc'):
            if target_prefix in line:
                inside_target = True
        elif line.startswith('</doc>'):
            inside_target = False
        else:
            if target_prefix in line:
                # this is the title
                continue
            if inside_target:
                print(line)


if __name__ == '__main__':
    main()
