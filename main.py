import os
import numpy as np
import matplotlib.pyplot as plt
import app.data_processing as dp
from models.cnn_multi_output import MultiOutputCNN
from models.cnn_single_output import SingleOutputCNN
from models.svm_model import svm_baseline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def cnn_multi(epochs=5, dimension=32):
    train_images = np.load('fingers/train_preprocessed.npy')
    test_images = np.load('fingers/test_preprocessed.npy')
    train_labels = np.load('labels/train_classes.npy')
    test_labels = np.load('labels/test_classes.npy')

    train_num_labels = train_labels[:, 0]
    train_hand_labels = train_labels[:, 1]
    test_num_labels = test_labels[:, 0]
    test_hand_labels = test_labels[:, 1]

    cnn = MultiOutputCNN(dimension)
    if os.path.isdir('models/saved/multi'):
        cnn.load()
    else:
        cnn.add_layers()
        cnn.add_outputs()
        cnn.model.summary()
        cnn.compile(loss={'num': 'sparse_categorical_crossentropy', 'hand': 'sparse_categorical_crossentropy'}, metrics={'num': 'accuracy', 'hand': 'accuracy'})
        cnn.train(train_images, {'num': train_num_labels, 'hand': train_hand_labels}, test_images, {'num': test_num_labels, 'hand': test_hand_labels}, epochs)
        cnn.save()

    cnn.evaluate(test_images, {'num': test_num_labels, 'hand': test_hand_labels})
    y_pred = cnn.model.predict(test_images)
    num_pred = y_pred[0].argmax(axis=-1)
    hand_pred = y_pred[1].argmax(axis=-1)
    print(num_pred)
    print(hand_pred)
    ConfusionMatrixDisplay(confusion_matrix(test_num_labels, num_pred),
                           display_labels=['0', '1', '2', '3', '4', '5']).plot()
    plt.show()
    ConfusionMatrixDisplay(confusion_matrix(test_hand_labels, hand_pred),
                           display_labels=['0', '1']).plot()
    plt.show()


def cnn_single(epochs=5, dimension=32):
    train_images = np.load('fingers/train_preprocessed.npy')
    test_images = np.load('fingers/test_preprocessed.npy')
    train_labels = np.load('labels/train_classes.npy')
    test_labels = np.load('labels/test_classes.npy')

    train_labels = train_labels[:, 2]
    test_labels = test_labels[:, 2]

    cnn = SingleOutputCNN(dimension)
    if os.path.isdir('models/saved/single'):
        cnn.load()
    else:
        cnn.add_layers()
        cnn.add_outputs()
        cnn.model.summary()
        cnn.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
        cnn.train(train_images, train_labels, test_images, test_labels, epochs)
        cnn.save()

    cnn.evaluate(test_images, test_labels)
    y_pred = cnn.model.predict(test_images)
    ConfusionMatrixDisplay(confusion_matrix(test_labels, y_pred),
                           display_labels=['0L', '0R', '1L', '1R', '2L', '2R',
                                           '3L', '3R', '4R', '4L', '5L', '5R']).plot()
    plt.show()
    print('testing')


def flatten_data():
    # Open the data and flatten it
    train_data = np.load("fingers/train_preprocessed.npy")
    test_data = np.load("fingers/test_preprocessed.npy")
    train_labels = np.load("labels/train_classes.npy")
    test_labels = np.load("labels/test_classes.npy")

    flat_test_data = np.array([d.flatten() for d in test_data])
    flat_train_data = np.array([d.flatten() for d in train_data])
    # print(flat_test_data.shape, flat_train_data.shape)
    return train_labels, test_labels, flat_train_data, flat_test_data


def svm_single_output(train_labels, test_labels, flat_train_data, flat_test_data):
    # Runs svm for 12 labels
    svm = svm_baseline()
    svm.fit(flat_train_data, train_labels[:, 2])
    svm.predict(flat_test_data)
    print(svm.report_scores(test_labels[:, 2]))
    # print confusion matrix
    ConfusionMatrixDisplay(confusion_matrix(test_labels[:, 2], svm.preds),
                           display_labels=['0L', '0R', '1L', '1R', '2L', '2R',
                                           '3L', '3R', '4R', '4L', '5L', '5R']).plot()
    plt.show()


def svm_num_fingers(train_labels, test_labels, flat_train_data, flat_test_data):
    # Runs svm for 6 labels
    svm = svm_baseline()
    svm.fit(flat_train_data, train_labels[:, 0])
    svm.predict(flat_test_data)
    print(svm.report_scores(test_labels[:, 0]))
    print(svm.report_scores(test_labels[:, 2]))
    # print confusion matrix
    ConfusionMatrixDisplay(confusion_matrix(test_labels[:, 2], svm.preds),
                           display_labels=['0', '1', '2', '3', '4', '5']).plot()
    plt.show()


def svm_handedness(train_labels, test_labels, flat_train_data, flat_test_data):
    # Runs svm for 2 labels
    svm = svm_baseline()
    svm.fit(flat_train_data, train_labels[:, 1])
    svm.predict(flat_test_data)
    print(svm.report_scores(test_labels[:, 1]))
    # print confusion matrix
    ConfusionMatrixDisplay(confusion_matrix(test_labels[:, 2], svm.preds),
                           display_labels=['L', 'R']).plot()
    plt.show()


def main():
    # print(dp.delete_old_files())
    print("size of test set: {0}".format(len(os.listdir('fingers/test'))))
    print("size of train set: {0}".format(len(os.listdir('fingers/train'))))
    dp.get_labels()
    # dp.pre_process_images()
    dp.noisify_images()
    dp.test()
    cnn_single()
    # cnn_multi()
    # train_labels, test_labels, flat_train_data, flat_test_data = flatten_data()
    # svm_single_output(train_labels, test_labels, flat_train_data, flat_test_data)
    # svm_num_fingers(train_labels, test_labels, flat_train_data, flat_test_data)
    # svm_handedness(train_labels, test_labels, flat_train_data, flat_test_data)


if __name__ == '__main__':
    main()
