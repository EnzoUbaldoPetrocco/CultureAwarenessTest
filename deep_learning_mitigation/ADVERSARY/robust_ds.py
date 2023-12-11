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


class Robust_ds(DSClass):
    def __init__(
        self, paths=None, greyscale=0, culture=0, flat=0, percent=0.1, model=None
    ):
        
        self.mitigation_dataset(paths, greyscale, flat)
        self.nineonedivision(culture, percent=percent)
        self.model = model
        self.augmented_dataset = []
        self.fast_gradient_method_augmented_dataset = []
        self.projected_gradient_decent_augmented_dataset = []


    def standard_augmentation(self, g_rot=0.2, g_noise=0.1):
        self.augmented_dataset = []
        data_augmentation = tf.keras.Sequential(
            [
                layers.RandomFlip("horizontal_and_vertical"),
                layers.RandomRotation(g_rot),
                layers.GaussianNoise(g_noise),
            ]
        )
        for culture in range(len(self.TestS)):
            cultureTS = []
            for X, y in self.TestS[culture]:
                X_augmented = data_augmentation(X)
                fig = plt.figure(figsize=(10, 7)) 
                #print(X)
                example = X[:, :, ::-1]
                example_augmented = X_augmented[:, :, ::-1]
                fig.add_subplot(1, 2, 1)
                #plt.imshow(example) 
                fig.add_subplot(1, 2, 2)
                #plt.imshow(example_augmented) 
                #plt.show()
                print(X==X_augmented)
                cultureTS.append((X_augmented, y))
            self.augmented_dataset.append(cultureTS)

    def fast_gradient_method_augmentation(self, eps=0.3):
        self.fast_gradient_method_augmented_dataset = []
        for culture in range(len(self.TestS)):
            cultureTS = []
            for X, y in self.TestS[culture]:
                X = X[None, ...]
                X_augmented = fast_gradient_method(
                    self.model, X, eps, np.inf
                )[0]
                cultureTS.append((X_augmented, y))
            self.fast_gradient_method_augmented_dataset.append(cultureTS)

    def projected_gradient_descent_augmentation(self, eps=0.3):
        self.projected_gradient_decent_augmented_dataset = []
        for culture in range(len(self.TestS)):
            cultureTS = []
            for X, y in self.TestS[culture]:
                X = X[None, ...]
                X_augmented = projected_gradient_descent(
                    self.model, X, eps, 0.01, 40, np.inf
                )[0]
                cultureTS.append((X_augmented, y))
            self.projected_gradient_decent_augmented_dataset.append(cultureTS)
