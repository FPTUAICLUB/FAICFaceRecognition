from detector import *


def main():
    detector = Detector(use_cuda=False)
    detector.run(0)


if __name__ == '__main__':
    main()