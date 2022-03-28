import os
import numpy as np
import app.data_processing as dp
from models.svm_model import svm_baseline


def flatten_data():
    # Open the data and flatten it
    train_data = np.load("fingers/train_preprocessed.npy")
    test_data = np.load("fingers/test_preprocessed.npy")
    train_labels = np.load("labels/train_classes.npy")
    test_labels = np.load("labels/test_classes.npy")

    flat_test_data = np.array([d.flatten() for d in test_data])
    flat_train_data = np.array([d.flatten() for d in train_data])

    return train_labels, test_labels, flat_train_data, flat_test_data


def svm_single_output(train_labels, test_labels, flat_train_data, flat_test_data):
    # Runs svm for 12 labels
    svm = svm_baseline()
    svm.fit(flat_train_data, train_labels[:, 2])
    svm.predict(flat_test_data)
    print(svm.report_scores(test_labels[:, 2]))


def svm_num_fingers(train_labels, test_labels, flat_train_data, flat_test_data):
    # Runs svm for 6 labels
    svm = svm_baseline()
    svm.fit(flat_train_data, train_labels[:, 0])
    svm.predict(flat_test_data)
    print(svm.report_scores(test_labels[:, 0]))


def svm_handedness(train_labels, test_labels, flat_train_data, flat_test_data):
    # Runs svm for 2 labels
    svm = svm_baseline()
    svm.fit(flat_train_data, train_labels[:, 1])
    svm.predict(flat_test_data)
    print(svm.report_scores(test_labels[:, 1]))


def main():
    # print(dp.delete_old_files())
    print("size of test set: {0}".format(len(os.listdir('fingers/test'))))
    print("size of train set: {0}".format(len(os.listdir('fingers/train'))))
    dp.get_labels()
    dp.pre_process_images()
    dp.test()

    train_labels, test_labels, flat_train_data, flat_test_data = flatten_data()
    # svm_single_output(train_labels, test_labels, flat_train_data, flat_test_data)
    # svm_num_fingers(train_labels, test_labels, flat_train_data, flat_test_data)
    svm_handedness(train_labels, test_labels, flat_train_data, flat_test_data)


if __name__ == '__main__':
    main()
