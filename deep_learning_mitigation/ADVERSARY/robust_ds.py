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
from cleverhans.tf2.utils import optimize_linear, compute_gradient

@tf.function
def my_compute_gradient(model_fn, loss_fn, x, y, targeted, culture=0):

    with tf.GradientTape() as g:
        g.watch(x)
        # Compute loss
        loss = loss_fn(y, model_fn(x)[culture])
        if (
            targeted
        ):  # attack is targeted, minimize loss of target label rather than maximize loss of correct label
            loss = -loss

    # Define gradient of loss wrt input
    grad = g.gradient(loss, x)
    return grad


def my_fast_gradient_method(
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
    plot = None
):
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
        y = tf.argmax(model_fn(x)[culture], 1)

    grad = my_compute_gradient(model_fn, loss_fn, x, y, targeted, culture=culture)

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
    return adv_x




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

    def standard_augmentation(self, g_rot=0.2, g_noise=0.1, g_bright=0.1):
        self.augmented_dataset = []
        data_augmentation = tf.keras.Sequential([
                 tf.keras.layers.RandomFlip("horizontal_and_vertical"),
                 tf.keras.layers.RandomRotation(g_rot),
                 tf.keras.layers.GaussianNoise(g_noise),
                 tf.keras.layers.RandomBrightness(g_bright)
            ])
        for culture in range(len(self.TestS)):
            cultureTS = []
            for X, y in self.TestS[culture]:
                X_augmented = data_augmentation(X, training=True)
                cultureTS.append((X_augmented, y))
            self.augmented_dataset.append(cultureTS)

    def fast_gradient_method_augmentation(self, eps=0.3):
        self.fast_gradient_method_augmented_dataset = []
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        for culture in range(len(self.TestS)):
            cultureTS = []
            for X, y in self.TestS[culture]:
                X = X[None, ...]
                y_i = y[1]*2-1
                y_i = tf.constant(y_i, dtype=tf.int32)
                y_i = y_i[None, ...]
                X_augmented = my_fast_gradient_method(
                    self.model, X, eps, np.inf, y=y_i, culture=culture, loss_fn=bce)
                #print(f'X=Xaugmented = {X==X_augmented}')
                #f, axarr = plt.subplots(2,1)
                #axarr[0].imshow(X[0][:, :, ::-1])
                #axarr[1].imshow(X_augmented[0][:, :, ::-1])
                #plt.show()
                cultureTS.append((X_augmented[0], y))
            self.fast_gradient_method_augmented_dataset.append(cultureTS)

    def projected_gradient_descent_augmentation(self, eps=0.3):
        self.projected_gradient_decent_augmented_dataset = []
        for culture in range(len(self.TestS)):
            cultureTS = []
            for X, y in self.TestS[culture]:
                X = X[None, ...]
                X_augmented = projected_gradient_descent(
                    self.model, X, eps, 0.01, 40, np.inf)
                cultureTS.append((X_augmented[0], y))
            self.projected_gradient_decent_augmented_dataset.append(cultureTS)
