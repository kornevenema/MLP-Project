import os
import numpy as np
import app.data_processing as dp
from models.cnn_single_output import SingleOutputCNN


def cnn_single():
    train_images = np.load('fingers/train_preprocessed.npy')
    test_images = np.load('fingers/test_preprocessed.npy')
    train_labels = np.load('labels/train_classes.npy')
    test_labels = np.load('labels/test_classes.npy')

    train_labels = train_labels[:, 2]
    test_labels = test_labels[:, 2]

    cnn = SingleOutputCNN(32)
    cnn.add_layers()
    cnn.add_outputs()
    cnn.model.summary()
    cnn.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
    cnn.train(train_images, train_labels, test_images, test_labels, 2)
    cnn.evaluate(test_images, test_labels)
    print('testing')


def main():
    # print(dp.delete_old_files())
    print("size of test set: {0}".format(len(os.listdir('fingers/test'))))
    print("size of train set: {0}".format(len(os.listdir('fingers/train'))))
    dp.get_labels()
    dp.pre_process_images()
    dp.test()
    cnn_single()


if __name__ == '__main__':
    main()
