import random

import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder


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
            fingers = []
            hand = []
            both = []
            for item in os.listdir(f"fingers/{t}"):
                item = item[:-4].split("_")[-1]
                both.append(item)
                fingers.append(list(item)[0])
                hand.append(list(item)[1])

            # transform fingers
            fingers_enc = LabelEncoder().fit(fingers)
            fingers = fingers_enc.transform(fingers)

            # transform hand
            hand_enc = LabelEncoder().fit(hand)
            hand = hand_enc.transform(hand)

            # transform both labels
            both_enc = LabelEncoder().fit(both)
            both = both_enc.transform(both)

            if t == "train":
                print("fingers mapping:    ", end=" ")
                print({l: i for i, l in enumerate(fingers_enc.classes_)})
                print("hand mapping:       ", end=" ")
                print({l: i for i, l in enumerate(hand_enc.classes_)})
                print("both labels mapping:", end=" ")
                print({l: i for i, l in enumerate(both_enc.classes_)})
            out = np.column_stack((fingers, hand, both))
            np.save(f'labels/{t}_classes.npy', out)
        else:
            print(f"{t} labels already exist")
            continue


def pre_process_images():
    # check if preprocessed images already exist
    for t in ["test", "train"]:
        if not os.path.isfile(f"fingers/{t}_preprocessed.npy"):
            print(f"preprocessing {t} images...")
            np.save(f"fingers/{t}_preprocessed.npy", np.array([
                np.array(
                    Image.open(f"fingers/{t}/{file}").resize((32, 32)).convert(
                        "RGB"))
                for file in os.listdir(f"fingers/{t}")
            ]))
        else:
            print(f"preprocessed {t} images already exist")
            continue


def noisify_images():
    # check if preprocessed images already exist
    for t in ["test", "train"]:
        if not os.path.isfile(f"fingers/{t}_preprocessed.npy"):
            print(f"preprocessing {t} images...")
            images = np.array([Image.open(f"fingers/{t}/{file}").convert("RGB")
                               for file in os.listdir(f"fingers/{t}")])
            blur = random.sample(range(0, len(images)), len(images) // 4)
            rotate = random.sample(range(0, len(images)), len(images) // 2)
            moldy = random.sample(range(0, len(images)), len(images) // 3)
            for n in blur:
                images[n] = images[n].filter(ImageFilter.GaussianBlur(5))
            for n in rotate:
                rand_num = random.randint(1, 3)
                if rand_num == 1:
                    images[n] = images[n].rotate(90)
                elif rand_num == 2:
                    images[n] = images[n].rotate(180)
                elif rand_num == 3:
                    images[n] = images[n].rotate(270)
            for n in moldy:
                images[n] = images[n].filter(
                    ImageFilter.Kernel((3, 3), (-1/ -5, 2/ -5, -2/ -5, 3/ -5, -5/ -5, 5/ -5, -7/ -5, 7/ -5, -7/ -5)))

            for i in range(10):
                plt.imshow(images[random.randrange(0, len(images), 1)])
                plt.show()
            np.save(f"fingers/{t}_preprocessed.npy",
                    np.array([np.array(im.resize((32, 32))) for im in images]))
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
    print("size of train images preprocessed: {0}".format(
        np.load("fingers/train_preprocessed.npy").shape))


def delete_old_files():
    os.remove('labels/test_classes.npy')
    os.remove('labels/train_classes.npy')
    os.remove('fingers/test_preprocessed.npy')
    os.remove('fingers/train_preprocessed.npy')
    return 'Files removed'
