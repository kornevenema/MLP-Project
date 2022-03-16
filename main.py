import os


def main():
    print("size of test set: {0}".format(len(os.listdir('data/test'))))
    print("size of train set: {0}".format(len(os.listdir('data/train'))))


if __name__ == '__main__':
    main()
