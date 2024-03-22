import os
import pathlib
import cv2
import random
import time
import tensorflow as tf
import numpy as np
from cleverhans.tf2.utils import optimize_linear
from matplotlib import pyplot as plt


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

    def get_images(self, path, n=1000):
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
            im = cv2.imread(str(i)) / 255
            im = im[..., ::-1]
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


## Preprocessing Class should:
# given a dataset it should perform standard data augmentation
# given a dataset and a model it should perform adversarial data augm
class PreprocessingClass:
    def classical_augmentation(self, X, g_rot=0.1, g_noise=0.1, g_bright=0.1, n=-1):
        """
        this function gets a set of images and return them augmented
        param: X: the set of images
        param: g_rot: is the gain of random rotation
        param: g_noise: is the gain of random noise
        param_ g_bright: is the gain of random brightness
        param_ n: number of images to perform the augmentation
        """
        if n <= 0 or n == None:
            n = len(X)
        X = X[0:n]

        X = tf.keras.layers.RandomFlip("horizontal_and_vertical")(X, training=True)
        X = tf.keras.layers.RandomRotation(g_rot)(X, training=True)
        X = tf.keras.layers.GaussianNoise(g_noise)(X, training=True)
        X_augmented = tf.keras.layers.RandomBrightness(g_bright / 5)(X, training=True)

        return np.asarray(X_augmented)

    def adversarial_augmentation(self, X, y, model, culture, eps=0.3):
        """
        adversarial_augmentation creates adversarial samples, i.e.
        samples that are created using samples in the dataset and the
        gradient for testing/enforce the robustness of a ML model

        :param X: samples to be used as starting point
        :param y: labels of the respective samples
        :param model: model used for computing the gradient
        :param culture: used because in our mitigation strategy we select the output using
        the culture from which the image derives
        :param eps: gain of the fast gradient method

        return: a set X_augmented of adversarial samples of the same size of X

        """
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        X_augmented = []
        for i in range(len(X)):
            x = X[i]
            y_i = y[i]
            if len(np.shape(y_i)) > 0:
                if np.shape(y_i)[0] > 1:
                    y_i = y_i[1]
            y_i = y_i * 2 - 1
            y_i = np.asarray([y_i], dtype=float)
            y_i = y_i[None, ...]
            X_augmented.append(
                self.my_fast_gradient_method(
                    model,
                    np.asarray(x[None, ...]),
                    eps,
                    np.inf,
                    y=y_i,
                    culture=culture,
                    loss_fn=bce,
                )
            )
        return X_augmented

    @tf.function
    def my_compute_gradient(self, model_fn, loss_fn, x, y, targeted, culture=0):
        """
        my_compute_gradient computes the gradient of a model

        :param model_fn: model with respect to compute the gradient
        :param loss_fn: loss function used for computing the gradient
        :param x: samples
        :param y: label of samples
        :param targeted: if targeted, minimize loss of target label rather than maximize loss of correct label
        :param culture: culture that selects the corresponding output

        :return gradient
        """
        with tf.GradientTape() as g:
            g.watch(x)
            # Compute loss
            yP = model_fn(x)
            if tf.size(yP) > 1:
                yP = yP[culture]
            else:
                yP = yP[None, ...]

            loss = loss_fn(y, yP)
            if (
                targeted
            ):  # attack is targeted, minimize loss of target label rather than maximize loss of correct label
                loss = -loss

        # Define gradient of loss wrt input
        grad = g.gradient(loss, x)
        return grad

    def my_fast_gradient_method(
        self,
        model_fn,
        x,
        eps,
        norm,
        loss_fn=None,
        clip_min=None,
        clip_max=None,
        y=None,
        targeted=False,
        sanity_checks=False,
        culture=0,
        plot=None,
    ):
        """
        Implementation of fast gradient method: the samples are moved against the
        gradient using an eps step

        :param model_fn: model w.r.t compute the gradient
        :param x: samples
        :param eps: gain of the step
        :param norm: type of norm to be applied to optimize perturbation
        :param loss_fn: loss function
        :param clip_min: minimum threshold for saturation
        :param clip_max: maximum threshold for saturation
        :param y: label of samples
        :param target:  if targeted, minimize loss of target label rather than maximize loss of correct label
        :param sanity_checks: if enable, checks for asserts
        :param culture: select the correct output in our Mitigation Strategy
        :param plot: if enabled, plot the adversarial sample

        """
        if norm not in [np.inf, 1, 2]:
            raise ValueError("Norm order must be either np.inf, 1, or 2.")

        if loss_fn is None:
            loss_fn = tf.nn.sparse_softmax_cross_entropy_with_logits

        asserts = []

        # If a data range was specified, check that the input was in that range
        if clip_min is not None:
            asserts.append(tf.math.greater_equal(x, clip_min))

        if clip_max is not None:
            asserts.append(tf.math.less_equal(x, clip_max))

        # cast to tensor if provided as numpy array
        x = tf.cast(x, tf.float32)

        if y is None:
            # Using model predictions as ground truth to avoid label leaking
            yf = model_fn(x)[:, culture]
            y = tf.argmax(yf, 1)
        grad = self.my_compute_gradient(
            model_fn, loss_fn, x, y, targeted, culture=culture
        )

        optimal_perturbation = optimize_linear(grad, eps, norm)

        if plot is not None:
            plt.imshow(optimal_perturbation[0])
            plt.show()

        # Add perturbation to original example to obtain adversarial example
        adv_x = x + optimal_perturbation

        # If clipping is needed, reset all values outside of [clip_min, clip_max]
        if (clip_min is not None) or (clip_max is not None):
            # We don't currently support one-sided clipping
            assert clip_min is not None and clip_max is not None
            adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)

        if sanity_checks:
            assert np.all(asserts)
        return np.asarray(adv_x[0])
