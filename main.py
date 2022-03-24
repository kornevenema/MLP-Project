import os
import numpy as np
import app.data_processing as dp
from models.svm_model import svm_baseline


def svm_single_output():
    # Opens labels and images and runs svm for 12 labels
    train_data = np.load("fingers/train_preprocessed.npy")
    test_data = np.load("fingers/test_preprocessed.npy")
    train_labels = np.load("fingers/test_preprocessed.npy")
    test_labels = np.load("fingers/train_preprocessed.npy")
    flat_test_data = np.array([d.flatten() for d in test_data])
    flat_train_data = np.array([d.flatten() for d in train_data])
    # print(flat_test_data.shape)
    svm = svm_baseline()
    # svm.fit(train_data, train_labels)


def main():
    # print(dp.delete_old_files())
    print("size of test set: {0}".format(len(os.listdir('fingers/test'))))
    print("size of train set: {0}".format(len(os.listdir('fingers/train'))))
    dp.get_labels()
    dp.pre_process_images()
    dp.test()
    svm_single_output()


if __name__ == '__main__':
    main()
