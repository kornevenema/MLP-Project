import os
import numpy as np
import app.data_processing as dp


def main():
    print("size of test set: {0}".format(len(os.listdir('fingers/test'))))
    print("size of train set: {0}".format(len(os.listdir('fingers/train'))))


if __name__ == '__main__':
    main()
