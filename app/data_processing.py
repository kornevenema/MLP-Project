import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os


def get_labels():
    # check if finger images are present
    if not os.path.isdir("fingers/test") or \
            not os.path.isdir("fingers/train"):
        print("train and test data is not present")
        return

    # check if labels folder already exist
    if not os.path.isdir("labels"):
        os.mkdir("labels")

    # create test and train labels
    for t in ["test", "train"]:
        if not os.path.isfile(f"labels/{t}_classes.npy"):
            print(f"getting labels from {t} images")
            labels = [list(item[:-4].split("_")[-1])
                      for item in os.listdir(f"fingers/{t}")]
            np.save(f'labels/{t}_classes.npy', labels)
        else:
            print(f"{t} labels already exist")
            continue


def pre_process_images():
    # check if preprocessed images already exist
    for t in ["test", "train"]:
        if not os.path.isfile(f"fingers/{t}_preprocessed.npy"):
            print(f"preprocessing {t} images...")
            np.save(f"fingers/{t}_preprocessed.npy", np.array([
                np.array(Image.open(f"fingers/{t}/{file}").resize((32, 32)))
                for file in os.listdir(f"fingers/{t}")
            ]))
        else:
            print(f"preprocessed {t} images already exist")
            continue


def test():
    print("size of test labels: {0}".format(
        np.load("labels/test_classes.npy").shape))
    print("size of train labels: {0}".format(
        np.load("labels/train_classes.npy").shape))
    print("size of test images preprocessed: {0}".format(
        np.load("fingers/test_preprocessed.npy").shape))
    print("size of test images preprocessed: {0}".format(
        np.load("fingers/train_preprocessed.npy").shape))


def delete_old_files():
    os.remove('labels/test_classes.npy')
    os.remove('labels/train_classes.npy')
    os.remove('fingers/test_preprocessed.npy')
    os.remove('fingers/train_preprocessed.npy')
    return 'Files removed'
