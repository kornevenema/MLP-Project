import os
import random
import numpy as np
import matplotlib.pyplot as plt
import app.data_processing as dp
from models.cnn_multi_output import MultiOutputCNN
from models.cnn_single_output import SingleOutputCNN
from models.svm_model import svm_baseline
from models.decision_model import DecisionTreeBaseline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score


def get_data_from_files(flatten=False, custom=False):
    """
    get the images and labels
    :param flatten: choose if you want to flatten data, only needed for baseline
    :return: return images and labels for the test and train data
    """
    # get data
    train_data = np.load("fingers/train_preprocessed.npy")
    train_labels = np.load("labels/train_classes.npy")
    if custom:
        test_data = np.load("fingers/customtest_preprocessed.npy")
        test_labels = np.load("labels/customtest_classes.npy")
    else:
        test_data = np.load("fingers/test_preprocessed.npy")
        test_labels = np.load("labels/test_classes.npy")

    if flatten:
        # flatten images
        flat_test_data = np.array([d.flatten() for d in test_data])
        flat_train_data = np.array([d.flatten() for d in train_data])

        # print(flat_test_data.shape, flat_train_data.shape)
        return train_labels, test_labels, flat_train_data, flat_test_data
    return train_labels, test_labels, train_data, test_data


def get_display_labels(t="both"):
    """
    get labels for the confusion matrix
    :param t: both, fingers, or hands
    :return: list
    """
    if t == "both":
        return ['0L', '0R', '1L', '1R', '2L', '2R',
                '3L', '3R', '4R', '4L', '5L', '5R']
    if t == "hands":
        return ["L", "R"]
    if t == "fingers":
        return [str(t) for t in range(0, 6)]


def cnn_multi(epochs=5, dimension=32):
    # get data
    train_labels, test_labels, train_images, test_images = get_data_from_files(custom=False)

    # separate numbers and fingers
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
        cnn.compile(loss={'num': 'sparse_categorical_crossentropy',
                          'hand': 'sparse_categorical_crossentropy'},
                    metrics={'num': 'accuracy', 'hand': 'accuracy'})
        cnn.train(train_images,
                  {'num': train_num_labels, 'hand': train_hand_labels},
                  test_images,
                  {'num': test_num_labels, 'hand': test_hand_labels}, epochs)
        cnn.save()

    cnn.evaluate(test_images,
                 {'num': test_num_labels, 'hand': test_hand_labels})
    y_pred = cnn.model.predict(test_images)
    num_pred = y_pred[0].argmax(axis=-1)
    hand_pred = y_pred[1].argmax(axis=-1)
    # print(num_pred)
    # print(hand_pred)
    ConfusionMatrixDisplay(confusion_matrix(test_num_labels, num_pred),
                           display_labels=get_display_labels("fingers")).plot()
    plt.title('Multi output CNN number of fingers Confusion Matrix')
    plt.show()
    ConfusionMatrixDisplay(confusion_matrix(test_hand_labels, hand_pred),
                           display_labels=get_display_labels("hands")).plot()
    plt.title('Multi output CNN handedness Confusion Matrix')
    plt.show()

    print("Macro avg f1 score for numbers: {}".format(
        f1_score(test_num_labels, num_pred, average="macro")))
    print("Macro avg f1 score for handedness: {}".format(
        f1_score(test_hand_labels, hand_pred, average="macro")))


def cnn_single(epochs=5, dimension=32):
    train_labels, test_labels, train_images, test_images = get_data_from_files(custom=False)

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
    y_pred = cnn.model.predict(test_images).argmax(axis=-1)
    ConfusionMatrixDisplay(confusion_matrix(test_labels, y_pred),
                           display_labels=get_display_labels()).plot()
    plt.title('Single output CNN Confusion Matrix')
    plt.show()
    # print('testing')
    print("Macro avg f1 score for single output: {}".format(
        f1_score(test_labels, y_pred, average="macro")))


def svm_single_output(train_labels, test_labels, flat_train_data,
                      flat_test_data):
    # Runs svm for 12 labels
    svm = svm_baseline()
    svm.fit(flat_train_data, train_labels[:, 2])
    svm.predict(flat_test_data)
    print(svm.report_scores(test_labels[:, 2]))
    # print confusion matrix
    ConfusionMatrixDisplay(confusion_matrix(test_labels[:, 2], svm.preds),
                           display_labels=get_display_labels()).plot()
    plt.title('12 class SVM Confusion Matrix')
    plt.show()

    print("Macro avg f1 score for single output SVM: {}".format(
        svm.f1_score(test_labels[:, 2])))


def svm_num_fingers(train_labels, test_labels, flat_train_data, flat_test_data):
    # Runs svm for 6 labels
    svm = svm_baseline()
    svm.fit(flat_train_data, train_labels[:, 0])
    svm.predict(flat_test_data)
    print(svm.report_scores(test_labels[:, 0]))
    # print confusion matrix
    ConfusionMatrixDisplay(confusion_matrix(test_labels[:, 0], svm.preds),
                           display_labels=get_display_labels("fingers")).plot()
    plt.title('SVM number of fingers Confusion Matrix')
    plt.show()

    print("Macro avg f1 score for numbers SVM: {}".format(
        svm.f1_score(test_labels[:, 0])))


def tree_num_fingers(train_labels, test_labels, flat_train_data, flat_test_data):
    # Runs svm for 2 labels
    tree = DecisionTreeBaseline()
    tree.fit(flat_train_data, train_labels[:, 0])
    tree.predict(flat_test_data)
    print(tree.report_scores(test_labels[:, 0]))
    # print confusion matrix
    ConfusionMatrixDisplay(confusion_matrix(test_labels[:, 0], tree.preds),
                           display_labels=get_display_labels("fingers")).plot()
    plt.title('Decision Tree number of fingers Confusion Matrix')
    plt.show()

    print("Macro avg f1 score for numbers DT: {}".format(f1_score(test_labels[:, 0], tree.preds, average="macro")))
    tree.print_tree()
    plt.show()


def tree_handedness(train_labels, test_labels, flat_train_data, flat_test_data):
    # Runs svm for 2 labels
    tree = DecisionTreeBaseline()
    tree.fit(flat_train_data, train_labels[:, 1])
    tree.predict(flat_test_data)
    print(tree.report_scores(test_labels[:, 1]))
    # print confusion matrix
    ConfusionMatrixDisplay(confusion_matrix(test_labels[:, 1], tree.preds),
                           display_labels=get_display_labels("hands")).plot()
    plt.title('Decision Tree handedness Confusion Matrix')
    plt.show()

    print("Macro avg f1 score for handedness DT: {}".format(f1_score(test_labels[:, 1], tree.preds, average="macro")))
    tree.print_tree()
    plt.show()


def main():
    # print(dp.delete_old_files())
    print("size of test set: {0}".format(len(os.listdir('fingers/test'))))
    print("size of train set: {0}".format(len(os.listdir('fingers/train'))))
    dp.get_labels()
    # dp.pre_process_images()
    dp.noisify_images()
    dp.test()
    # cnn_single(epochs=20)
    cnn_multi(epochs=20)
    # train_labels, test_labels, flat_train_data, flat_test_data = \
    #     get_data_from_files(True, custom=False)
    # svm_single_output(train_labels, test_labels, flat_train_data,
    #                   flat_test_data)
    # svm_num_fingers(train_labels, test_labels, flat_train_data, flat_test_data)
    # tree_num_fingers(train_labels, test_labels, flat_train_data, flat_test_data)
    # tree_handedness(train_labels, test_labels, flat_train_data, flat_test_data)


if __name__ == '__main__':
    main()
