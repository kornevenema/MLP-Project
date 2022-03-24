import os
import numpy as np
import app.data_processing as dp


def main():
    # print(dp.delete_old_files())
    print("size of test set: {0}".format(len(os.listdir('fingers/test'))))
    print("size of train set: {0}".format(len(os.listdir('fingers/train'))))
    dp.get_labels()
    dp.pre_process_images()
    dp.test()


if __name__ == '__main__':
    main()
