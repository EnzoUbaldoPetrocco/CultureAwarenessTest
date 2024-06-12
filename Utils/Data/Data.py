#!/usr/bin/env python
__author__ = "Enzo Ubaldo Petrocco"
import os
import pathlib
import cv2
import random
import time
import numpy as np


## DataClass should:
# Collect all images of a given directory and prepare the labels
# Hp1: we have some cultures, which should be initially separated
# Hp2: we are dealing with a classification task
# The structure of the dataset is:
# List per Cultures
# List per Label
# Then we need to build a function that prepares the dataset
# in different contexts (shallow, deep, or )
class DataClass:
    """DataClass is a util class for managing data


    DataClass initialization contains only the dataset got from
    the folder.
    Time by time a function is called in order to prepare
    Tr, V and Te division inside variables (X,y), (Xv, yv)
    and (Xt, yt) respectively.

    Attributes:
        dataset: is a list of datasets per culture. each dataset
        contains a list of data subdivided per labels in the form (X,y)
        X contains the image in RGB
        y contains the label in the form (culture, class)
    """

    def __init__(self, paths) -> None:
        """
        init function gets the images from the folder
        images are stored inside self.dataset variable

        :param paths: contains the paths from which the program gets
        the images
        :return None
        """
        self.dataset = []
        for j, path in enumerate(paths):
            labels = self.get_labels(path)
            imgs_per_culture = []
            for i, label in enumerate(labels):
                images = self.get_images(path + "/" + label)
                X = []
                for k in range(len(images)):
                    X.append([images[k], [j, i]])
                # print(f"Culture is {j}, label is {i}")
                # plt.imshow(images[0])
                # plt.show()
                imgs_per_culture.append(X)
                del images
                del X
            self.dataset.append(imgs_per_culture)

    def get_labels(self, path):
        """
        get_labels returns a list of the labels in a directory

        :param path: directory in which search of the labels
        :return list of labels
        """
        dir_list = []
        for file in os.listdir(path):
            d = os.path.join(path, file)
            if os.path.isdir(d):
                d = d.split("\\")
                if len(d) == 1:
                    d = d[0].split("/")
                d = d[-1]
                dir_list.append(d)
        return dir_list

    def get_images(self, path, n=1000, rescale=True):
        """
        get_images returns min(n, #images contained in a directory)

        :param path: directory in which search for images
        :param n: maximum number of images

        :return list of images
        """
        images = []
        types = ("*.png", "*.jpg", "*.jpeg")
        paths = []
        for typ in types:
            paths.extend(pathlib.Path(path).glob(typ))
        paths = paths[0 : min(len(paths), n)]
        for i in paths:
            im = cv2.imread(str(i)) 
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            if rescale:
                im = im  /255
            im = np.moveaxis(im, -1, 0)
            #im = np.transpose(im)
            images.append(im)
        return images

    # STANDARD ML
    # Standard ML does not need culture information
    # Standard ML does not need culture division for Tr and V sets
    def prepare(
        self,
        standard,
        culture,
        percent=0,
        shallow=0,
        val_split=0.2,
        test_split=0.2,
        n=1000,
    ):
        """
        this function prepares time by time the sets for training
        and evaluating the models
        :standard standard ML does not need culture information in labels
        :param culture: states which is the majority culture
        :param percent: indicates which percentage of minority culture is
        put inside Tr and V sets
        :param shallow: if shallow learning images are converted
        to Greyscale and then flattened
        :param val_split: indicates the validation percentage with respect
        to the sum between Tr and V sets
        :param test_split: indicates the test percentage with respect to the
        whole dataset
        :param n: number of images for each class and culture
        :return None
        """
        random.seed(time.time_ns())

        self.X = []
        self.y = []
        self.Xv = []
        self.yv = []
        self.Xt = []
        self.yt = []
        for c, cDS in enumerate(self.dataset):
            cultureXt = []
            cultureyT = []
            for lb, lDS in enumerate(cDS):
                random.shuffle(lDS)
                Xds = []
                yds = []

                for img, label in lDS:
                    if standard:
                        label = int(label[1])
                    if shallow:
                        img = img[0::]
                        img = img.flatten()
                    Xds.append(img)
                    yds.append(label)
                nt = int(n * test_split)
                cultureXt.extend(Xds[0:nt])
                cultureyT.extend(yds[0:nt])

                Xds = Xds[nt : n - 1]
                yds = yds[nt : n - 1]
                if percent != 0:
                    if c != culture:
                        Xds = Xds[0 : int(percent * len(Xds))]
                        yds = yds[0 : int(percent * len(yds))]
                    nv = int(val_split * len(Xds))
                    self.Xv.extend(Xds[0:nv])
                    self.yv.extend(yds[0:nv])
                    self.X.extend(Xds[nv : len(Xds)])
                    self.y.extend(yds[nv : len(yds)])
                else:
                    if c == culture:
                        nv = int(val_split * len(Xds))
                        self.Xv.extend(Xds[0:nv])
                        self.yv.extend(yds[0:nv])
                        self.X.extend(Xds[nv : len(Xds)])
                        self.y.extend(yds[nv : len(yds)])
                del Xds
                del yds
            self.Xt.append(cultureXt)
            self.yt.append(cultureyT)

    def clear(self):
        """
        clear empty all the dataset divisions
        """
        self.X = None
        self.y = None
        self.Xv = None
        self.yv = None
        self.Xt = None
        self.yt = None
        del self.X
        del self.y
        del self.Xv
        del self.yv
        del self.Xt
        del self.yt


    