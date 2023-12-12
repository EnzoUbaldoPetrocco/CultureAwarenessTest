import sys

sys.path.insert(1, "../../../")
from DS.ds import DSClass
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import cv2
from matplotlib import pyplot as plt
from tensorflow.keras.layers.experimental import preprocessing
import keras



class Robust_ds(DSClass):
    def __init__(
        self, paths=None, greyscale=0, culture=0, flat=0, percent=0.1, model=None, lamb=0
    ):
        
        self.mitigation_dataset(paths, greyscale, flat)
        self.nineonedivision(culture, percent=percent)
        self.model = model
        self.augmented_dataset = []
        self.fast_gradient_method_augmented_dataset = []
        self.projected_gradient_decent_augmented_dataset = []
        self.lamb = lamb



    def standard_augmentation(self, g_rot=0.2, g_noise=0.1):
        self.augmented_dataset = []

        data_augmentation = keras.Sequential([
                 keras.layers.RandomFlip("horizontal_and_vertical"),
                 keras.layers.RandomRotation(g_rot),
                 keras.layers.GaussianNoise(g_noise)
            ])
        for culture in range(len(self.TestS)):
            cultureTS = []
            for X, y in self.TestS[culture]:
                X_augmented = data_augmentation(X, training=True)


                cultureTS.append((X_augmented, y))
            self.augmented_dataset.append(cultureTS)

    def fast_gradient_method_augmentation(self, eps=0.3):
        def loss(labels, logits):
            print(np.shape(logits))
            weights1 = self.model.layers[len(self.model.layers) - 1].kernel
            weights2 = self.model.layers[len(self.model.layers) - 2].kernel
            weights3 = self.model.layers[len(self.model.layers) - 3].kernel
            mean = tf.math.add(weights1, weights2)
            mean = tf.math.add(mean, weights3)
            mean = tf.multiply(mean, 1 / 3)
            mean = tf.multiply(mean, self.lamb)
            if culture == 0:
                dist = tf.norm(weights1 - mean, ord="euclidean")
            if culture == 1:
                dist = tf.norm(weights2 - mean, ord="euclidean")
            if culture == 2:
                dist = tf.norm(weights3 - mean, ord="euclidean")
            dist = tf.multiply(dist, dist)
            # dist12 = tf.norm(weights1-weights2, ord='euclidean')
            # dist13 = tf.norm(weights1-weights3, ord='euclidean')
            # dist23 = tf.norm(weights2-weights3, ord='euclidean')
            # dist = tf.math.add(dist12, dist13)
            # dist = tf.math.add(dist, dist23)
            # dist = tf.multiply(tf.multiply(dist,dist) , .lamb)
            loss = tf.keras.losses.binary_crossentropy(labels[1], logits)
            res = tf.math.add(loss, dist)
            mask = tf.reduce_all(tf.equal(labels[0], culture))
            if not mask:
                return 0.0
            else:
                return res

        self.fast_gradient_method_augmented_dataset = []
        for culture in range(len(self.TestS)):
            cultureTS = []
            for X, y in self.TestS[culture]:
                X = X[None, ...]
                print(np.shape(y))
                X_augmented = fast_gradient_method(
                    self.model, X, eps, np.inf, loss_fn = loss, y=[y[0], y[1]]
                )

                cultureTS.append((X_augmented[0], y))
            self.fast_gradient_method_augmented_dataset.append(cultureTS)

    def projected_gradient_descent_augmentation(self, eps=0.3):
        def loss(labels, logits):
            weights1 = self.model.layers[len(self.model.layers) - 1].kernel
            weights2 = self.model.layers[len(self.model.layers) - 2].kernel
            weights3 = self.model.layers[len(self.model.layers) - 3].kernel
            mean = tf.math.add(weights1, weights2)
            mean = tf.math.add(mean, weights3)
            mean = tf.multiply(mean, 1 / 3)
            mean = tf.multiply(mean, self.lamb)
            if culture == 0:
                dist = tf.norm(weights1 - mean, ord="euclidean")
            if culture == 1:
                dist = tf.norm(weights2 - mean, ord="euclidean")
            if culture == 2:
                dist = tf.norm(weights3 - mean, ord="euclidean")
            dist = tf.multiply(dist, dist)
            # dist12 = tf.norm(weights1-weights2, ord='euclidean')
            # dist13 = tf.norm(weights1-weights3, ord='euclidean')
            # dist23 = tf.norm(weights2-weights3, ord='euclidean')
            # dist = tf.math.add(dist12, dist13)
            # dist = tf.math.add(dist, dist23)
            # dist = tf.multiply(tf.multiply(dist,dist) , .lamb)
            loss = tf.keras.losses.binary_crossentropy(labels[1], logits)
            res = tf.math.add(loss, dist)
            mask = tf.reduce_all(tf.equal(labels[0], culture))
            if not mask:
                return 0.0
            else:
                return res
        self.projected_gradient_decent_augmented_dataset = []
        for culture in range(len(self.TestS)):
            cultureTS = []
            for X, y in self.TestS[culture]:
                X = X[None, ...]
                X_augmented = projected_gradient_descent(
                    self.model, X, eps, 0.01, 40, np.inf, loss_fn = loss, y=[y[0], y[1]]
                )
                cultureTS.append((X_augmented[0], y))
            self.projected_gradient_decent_augmented_dataset.append(cultureTS)
