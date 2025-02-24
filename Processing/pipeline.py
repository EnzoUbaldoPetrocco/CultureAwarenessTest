#!/usr/bin/env python
__author__ = "Enzo Ubaldo Petrocco"

import os
import pathlib
import cv2
import keras
import tensorflow as tf
from keras import layers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore
import numpy as np
from matlotlib import pyplot as plt
from random import randint


def get_dataset_path(search_root):
    """
    get main path to the dataset
    """
    for root, subdirs, _ in os.walk(search_root):
        for d in subdirs:
            if d == "FINALDS":
                return os.path.abspath(os.path.join(root, d))


def get_culture_paths(ds_pt, lamp):
    """
    get paths to every dataset culture
    """
    if lamp:
        return [
            ds_pt + "/lamps/chinese/100/RGB",
            ds_pt + "/lamps/french/100/RGB",
            ds_pt + "/lamps/turkish/100/RGB",
        ]
    return [
        ds_pt + "/carpets_stretched/indian/100/RGB",
        ds_pt + "/carpets_stretched/japanese/100/RGB",
        ds_pt + "/carpets_stretched/scandinavian/100/RGB",
    ]


class Pipeline:
    """
    Class implementing steps for machine learning models under
    underrepresented constrains.
    """

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
        print(dir_list)
        return dir_list

    def get_images(self, path, culture, label, n=1000):
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
            im = im[..., ::-1]
            if self.augment:
                images.append(im)
                im = self.data_augmentation(im)
            images.append(im, [culture, label])
        return images

    def build_dataset(self):
        """Build the dataset using structure: [self.n_cultures, n_samples, 2]
        Last two channels are images [size,size,3] and labels [n_cultures + 1], respectively

        :return None

        """
        # dataset is [self.n_cultures, n_samples, 2]

        for j, path in enumerate(self.culture_paths):
            c = np.zeros(self.n_cultures)
            c[j] = 1
            labels = self.get_labels(path)
            imgs_per_culture = []
            for i, label in enumerate(labels):
                imgs_per_culture.append(
                    self.get_images(path + "/" + label, c, i)
                )  # j is culture, i is label
            self.dataset.append(imgs_per_culture)

    def __init__(
        self,
        search_root="../../",
        lamp=True,
        save_root="./",
        verbose_param=True,
        shape=100,
        n_cultures=3,
        majority_culture=0,
        pu=0.05,
        proportions=None,
        oversampling=False,
        os_n=100,
        adversarial=False,
        epsilon=0.1,
        alpha=0.0002,
        num_iter=800,
        class_div=False,
        augment=False,
        g=0.01,
    ):
        """
        Initialize the class ML Pipeline

        :param search_root: directory in which search for dataset
        :param lamp: use lamp (True) or carpet (False) dataset
        :param save_root: directory in which save results or images
        :param shape: dimension of the images in dataset
        :param n_cultures: number of cultures contained in dataset
        :param majority culture: index of majority culture (consider alphabetic order)
        :param pu: percentage of images from minority cultures to take from their datasets
        :param proportions: splitting procedure proportions (LS, VS, TS)
        :param oversampling: use oversampling as mitigation strategy
        :param os_n: how many images add for implementing oversampling mitigation strategy
        :param adversarial: use adversarial sampling as mitigation strategy
        :param epsilon: maximum distance from original image to adversarial sample when implementing projected gradient descent attack
        :param alpha: step size in projected gradient descent attack
        :param num_iter: number of interation in PGD
        :param class_div: discriminator is trained considering label (True) or not (False)
        :param augment: use online and offline data augmentation strategies (Gaussian Noise, Flipping, ...)
        :param g: gain in some data augmentation strategy transformations

        :return None
        """
        # algorithm parameters
        self.lamp = lamp
        self.save_root = save_root
        self.verbose_param = verbose_param
        self.shape = shape
        self.n_cultures = n_cultures
        self.majority_culture = majority_culture
        self.pu = pu
        if len(proportions) == 3:
            self.proportions = proportions
        else:
            self.proportions = [0.7, 0.2, 0.1]
        # oversampling techniques
        self.os = oversampling
        self.os_n = os_n
        # adversarial parameters
        self.adversarial = adversarial
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iter = num_iter
        self.class_div = class_div
        # standard augmentation parameters
        self.augment = augment
        self.g = g

        # Defining temporarily void attributes:
        self.base_model = None
        self.model = None
        self.callbacks = None
        self.dataset = []
        self.base_path = None

        if self.augment:
            self.data_augmentation = keras.Sequential(
                [
                    keras.layers.Rescaling(scale=1.0 / 255),
                    layers.RandomFlip("horizontal"),
                    layers.RandomRotation(0.01),
                    layers.GaussianNoise(self.g),
                    keras.layers.RandomBrightness(0.01),
                    layers.RandomZoom(self.g, self.g),
                    layers.Resizing(self.shape, self.shape),
                    keras.layers.Rescaling(scale=255.0),
                ]
            )
        ds_pt = get_dataset_path(search_root)
        self.culture_paths = get_culture_paths(ds_pt, self.lamp)

    def build_model(self, n_outs, n_dropout, monitor_val):
        """
        Build a classification model base on ResNet using
        ImageNet weights and a sequence of GlobalAveragePooling2D,
        Dropout and Dense layer as head.
        If self.augmentation = True, it includes a layer of random transformations
        as preprocessing layer that is used during learning procedure.
        By default, ResNet is used only for inference, layers are freezed.

        :param n_outs: number of outs of the model
        :param n_drouput: dropout rate in Dropout layer
        :param monitor_val: metric to monitor for implementing overfitting
        regularization strategies

        :return None

        """
        # MODEL IMPLEMENTATION
        self.base_model = keras.applications.ResNet50V2(
            weights="imagenet",  # Load weights pre-trained on ImageNet.
            input_shape=self.shape,
            include_top=False,
        )  # Do not include the ImageNet classifier at the top.

        # Freeze the base_model
        self.base_model.trainable = False

        # Create  model on top
        inputs = keras.Input(shape=self.shape)

        scale_layer = keras.layers.Rescaling(scale=1 / 255.0)
        if self.augment:
            x = self.data_augmentation(inputs)  # Apply random data augmentation
            x = scale_layer(x)
        else:
            x = scale_layer(inputs)

        # The base model contains batchnorm layers. We want to keep them in inference mode
        # when we unfreeze the base model for fine-tuning, so we make sure that the
        # base_model is running in inference mode here.
        x = self.base_model(x, training=False)
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dropout(n_dropout)(x)  # Regularize with dropout
        if n_outs > 1:
            outputs = keras.layers.Dense(n_outs, activation="softmax")(x)
        else:
            outputs = keras.layers.Dense(n_outs, activation="sigmoid")(x)
        self.model = keras.Model(inputs, outputs)

        lr_reduce = ReduceLROnPlateau(
            monitor=monitor_val,
            factor=0.2,
            patience=5,
            verbose=self.verbose_param,
            min_lr=1e-9,
        )
        early = EarlyStopping(
            monitor=monitor_val,
            min_delta=0.001,
            patience=10,
            verbose=self.verbose_param,
            mode="auto",
        )
        self.callbacks = [early, lr_reduce]

    def train(self, lr, loss, metric, epochs, ls, vs, fine_lr, fine_epochs, batch_size):
        """
        Train the model and saves it in self.model. Initially, backbone is freezed,
        then a fine tuning procedure is applied.

        :param lr: learning rate
        :param loss: loss function
        :param metric: metric
        :param epochs: number of epochs
        :param ls: learning set
        :param vs: validation set
        :param fine_lr: learning rate during fine tuning procedure
        :param batch_size: batch size

        :return History: track of metrics and loss during epochs

        """
        ls = (
            tf.data.Dataset.from_tensor_slices(ls)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
            .cache()
        )
        vs = (
            tf.data.Dataset.from_tensor_slices(vs)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
            .cache()
        )

        # self.model.summary()
        # MODEL TRAINING
        self.model.compile(
            optimizer=keras.optimizers.Adam(lr),
            loss=loss,
            metrics=[metric],
        )

        self.model.fit(
            ls,
            epochs=epochs,
            validation_data=vs,
            verbose=self.verbose_param,
            callbacks=self.callbacks,
            shuffle=True,
        )

        # FINE TUNING
        self.base_model.trainable = True
        # self.model.summary()

        self.model.compile(
            optimizer=keras.optimizers.Adam(fine_lr),  # Low learning rate
            loss=loss,
            metrics=[metric],
        )

        history = self.model.fit(
            ls,
            epochs=fine_epochs,
            validation_data=vs,
            verbose=self.verbose_param,
            callbacks=self.callbacks,
            shuffle=True,
        )
        keras.backend.clear_session()
        return history

    def adversarial_training(self, ls, vs):
        """
        Calls model selection strategy for saving in self.model adversarial
        model.

        :param ls: learning set
        :param vs: validation set

        :return None
        """
        n_out = self.n_cultures
        loss = keras.losses.CategoricalCrossentropy(from_logits=True)
        metric = keras.metrics.CategoricalAccuracy()

        self.model_selection(ls, vs, n_out, loss, metric)

    def generate_adversarial_image_pgd(self, img, lbl, model):
        """Parameters:
        - model: the target model to attack.
        - x: the input images (batch).
        - y: the true labels corresponding to x.
        - epsilon: the maximum perturbation amount.
        - alpha: the step size for each iteration.
        - num_iter: the number of iterations for the PGD attack.

        Returns:
        - x_adv: the adversarial examples generated from x.
        """
        img = tf.expand_dims(img, axis=0)
        lbl = tf.expand_dims(lbl, axis=0)
        img = tf.convert_to_tensor(img)
        lbl = tf.convert_to_tensor(lbl)

        x_adv = tf.identity(img)  # Start from the original input

        for _ in range(self.num_iter):
            with tf.GradientTape() as tape:
                tape.watch(x_adv)
                prediction = model(x_adv)
                loss = keras.losses.categorical_crossentropy(lbl, prediction)

            # Get the gradients of the loss w.r.t. the input image.
            gradients = tape.gradient(loss, x_adv)

            # Perform the gradient ascent step
            perturbations = self.alpha * tf.sign(gradients)
            x_adv = x_adv / 255.0 + perturbations

            # Project the perturbation onto the epsilon ball
            x_adv = (
                tf.clip_by_value(
                    x_adv, img / 255.0 - self.epsilon, img / 255.0 + self.epsilon
                )
                * 255.0
            )
            x_adv = tf.clip_by_value(
                x_adv, 0, 255.0
            )  # Ensure the pixel values are still valid
        return x_adv

    def adversarial_samples(self, adversarial_model, samples, labels):
        """
        From an adversarial model, samples and labels generates adversarial sampels

        :param adversarial model: model used as discriminator
        :param samples: samples to be transformed
        :param labels: labels to be used for getting the model closer to the boundary

        :return adv_samples: adversarial samples

        """
        adv_samples = []
        for sample, label in (samples, labels):
            adv_samples.append(
                self.generate_adversarial_image_pgd(sample, label, adversarial_model)
            )

        return adv_samples

    def splitting_procedure(self):
        """
        Split dataset in learning, validation and test set

        :return ls: learning set
        :return vs: validation set
        :return ts: test set
        """
        x = []
        y = []
        xv = []
        yv = []
        xt = []
        yt = []

        for c, cds in enumerate(self.dataset):
            # Shuffle data
            cds = np.asarray(cds)
            perm = np.random.permutation(len(cds))
            cds = cds[perm]
            indeces = len(cds) * self.proportions
            if c != self.majority_culture:
                indeces = self.pu * indeces

            x.extend(list(cds[0 : indeces[0]][:, 0]))
            y.extend(list(cds[0 : indeces[0]][:, 1]))
            xv.extend(list(cds[indeces[0] : indeces[1]][:, 0]))
            yv.extend(list(cds[indeces[0] : indeces[1]][:, 1]))
            # Append because I want to keep them separated
            xt.append(list(cds[indeces[1] : indeces[2]][:, 0]))
            yt.append(list(cds[indeces[1] : indeces[2]][:, 1]))

            if self.os and c != self.majority_culture:
                x.extend(list(cds[0 : self.os_n][:, 0]))
                y.extend(list(cds[0 : self.os_n][:, 1]))

            cds = list(cds)

        ls = [x, y]
        vs = [xv, yv]
        ts = [xt, yt]
        return ls, vs, ts

    def preprocessing(self):
        """
        Function used for preprocessing dataset. Saves the performance of the
        discriminator in case of adversarial training

        :return ls: new learning set
        :return vs: new validation set
        :return ts: test set

        """
        self.dataset = []
        self.build_dataset()

        ls, vs, ts = self.splitting_procedure()

        if self.adversarial:
            samples = ls[0]
            labels = ls[1][0 : self.n_cultures]
            classes = ls[1][self.n_cultures]
            samples_v = vs[0]
            labels_v = vs[1][0 : self.n_cultures]
            classes_v = vs[1][self.n_cultures]
            if self.class_div:
                for i in range(2):
                    indeces = np.where(np.any(classes == i, axis=0))
                    indeces_v = np.where(np.any(classes_v == i, axis=0))
                    self.adversarial_training(
                        (
                            list(map(lambda i: samples[i], indeces)),
                            list(map(lambda i: labels[i], indeces)),
                        ),
                        (
                            list(map(lambda i: samples_v[i], indeces_v)),
                            list(map(lambda i: labels_v[i], indeces_v)),
                        ),
                    )
                    adversarial_samples = self.adversarial_samples(
                        self.model,
                        list(map(lambda i: samples[i], indeces)),
                        list(map(lambda i: labels[i], indeces)),
                    )
                    ls.extend([adversarial_samples, labels])
                    for c in range(self.n_cultures):
                        err = self.error_estimation(ts[c])
                    self.save_results(err, True, i)
            else:
                self.adversarial_training((samples, labels), (samples_v, labels_v))
                adversarial_samples = self.adversarial_samples(
                    self.model, samples, labels
                )
                ls.extend([adversarial_samples, labels])
                for c in range(self.n_cultures):
                    err = self.error_estimation(ts[c])
                self.save_results(err, True)

        return ls, vs, ts

    def model_selection(self, ls: list, vs, n_outs, loss, metric):
        """
        General function for implementing model selection procedure

        :param ls: learning set
        :param vs: validation set
        :param n_outs: number of classes
        :param loss: loss function
        :param metric: metric function

        :return None
        """
        fine_epochs = 5
        opt_hyper = {
            "bs": 0,
            "ne": 0,
            "lr": 0,
            "fine_lr": 0,
            "n_dropout": 0,
            "loss": np.inf,
        }
        monitor_val = "val_loss"
        for batch_size in np.logspace(0, 6, 7, base=2):
            for ne in np.logspace(1, 1.5, 3):
                ne = int(ne)
                for lr in np.logspace(-5, -3, 3):
                    for fine_lr in np.logspace(-6, -5, 2):
                        for n_dropout in [0.3, 0.4]:
                            self.build_model(n_outs, n_dropout, monitor_val)
                            history = self.train(
                                lr,
                                loss,
                                metric,
                                ne,
                                ls,
                                vs,
                                fine_lr,
                                fine_epochs,
                                batch_size,
                            )
                            val_metric = history.history[monitor_val][-1]
                            if val_metric <= np.inf:
                                opt_hyper["bs"] = batch_size
                                opt_hyper["ne"] = ne
                                opt_hyper["lr"] = lr
                                opt_hyper["fine_lr"] = fine_lr
                                opt_hyper["n_dropout"] = n_dropout

        monitor_val = "loss"
        self.build_model(n_outs, opt_hyper["n_dropout"], monitor_val)
        self.train(
            opt_hyper["lr"],
            loss,
            metric,
            opt_hyper["ne"],
            ls.extend(vs),
            None,
            opt_hyper["fine_lr"],
            fine_epochs,
            opt_hyper["bs"],
        )

    def error_estimation(self, ts):
        """
        General function for estimating the error using evaluate method from model

        :param ts: test set (images, labels)

        :return results: results obtained from function evaluate
        """
        results = self.model.evaluate(ts[0], ts[1], batch_size=16)
        return results

    def save_results(self, results, discriminator=False, pth_append=""):
        """
        Save results in a file

        :param results: results to be saved in the file
        :param: discriminator: boolean for knowing if the results come from a discriminator
        or not
        :param pth_append: path to the file

        :return None
        """
        self.build_path(discriminator=discriminator)
        with open(self.base_path + pth_append + "res.txt", "a", encoding="utf-8") as hs:
            hs.write(str(results) + "\n")
            hs.close()

    def build_path(self, discriminator=False):
        """
        Build path depending on: discriminator, oversampling, lamp, majority culture,
        percentage, adversarial and augmentation parameters

        :param discriminator: boolean that states if the path is about a discriminator model
        """
        self.base_path = self.save_root
        if discriminator:
            self.base_path = self.base_path + "/DISCR/"

        if self.os:
            self.base_path = self.base_path + "/OS/"
        else:
            self.base_path = self.base_path + "/NOOS/"
        if self.lamp:
            self.base_path += "/LAMP/"
        else:
            self.base_path += "/CARPET/"

        self.base_path += f"{self.majority_culture}/"

        self.base_path = self.base_path + str(self.pu) + "/"

        if self.augment:
            if self.adversarial:

                aug = f"TOTAUG/g={self.g}/eps={self.epsilon}/"
                if self.class_div:
                    aug = aug + "/CLSDIV/"
                else:
                    aug = aug + "/NOCLSDIV/"
            else:
                aug = f"STDAUG/g={self.g}/"
        else:
            if self.adversarial:
                aug = f"AVD/eps={self.epsilon}/"
                if self.class_div:
                    aug = aug + "/CLSDIV/"
                else:
                    aug = aug + "/NOCLSDIV/"

            else:
                aug = "NOAUG/"

        self.base_path = self.base_path + aug

    def plot_std_images(self):
        """
        Plot standard augmentaton images call after init function
        """
        self.dataset = []
        self.build_dataset()

        def augment_image(g, image):
            data_augmentation = keras.Sequential(
                [
                    keras.layers.Rescaling(scale=1.0 / 255),
                    layers.RandomFlip("horizontal"),
                    layers.RandomRotation(0.01),
                    layers.GaussianNoise(g),
                    keras.layers.RandomBrightness(0.01),
                    layers.RandomZoom(g, g),
                    layers.Resizing(self.shape, self.shape),
                    keras.layers.Rescaling(scale=255.0),
                ]
            )
            return data_augmentation(image)

        num_cols = 2
        num_rows = 3
        gs = np.logspace(-4, -1, num_cols * num_rows)
        for i, cds in enumerate(self.dataset):
            random_image = cds[randint(0, len(cds) - 1)]

            plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0))
            for row in range(num_rows):
                for col in range(num_cols):
                    index = row * num_cols + col
                    plt.subplot(num_rows, num_cols, index + 1)
                    plt.imshow(augment_image(gs[index], random_image[0]))
                    plt.axis("off")
                    # plt.imsave(f"./Sample{index}", images[index])
            plt.tight_layout()
            plt.savefig(self.save_root + "/LAMP=" + self.lamp + "/CULTURE=" + i)
            plt.show()
            plt.close()

    def plot_adv_images(self):
        """
        Plot adversarial augmentaton images call after init function
        """
        self.dataset = []
        self.build_dataset()

        ls, vs, _ = self.splitting_procedure()
        self.adversarial_training(ls, vs)

        num_cols = 2
        num_rows = 3
        eps = np.logspace(-4, -1, num_cols * num_rows)
        for i, cds in enumerate(self.dataset):
            random_image = cds[randint(0, len(cds) - 1)]

            plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0))
            for row in range(num_rows):
                for col in range(num_cols):
                    index = row * num_cols + col
                    plt.subplot(num_rows, num_cols, index + 1)
                    self.epsilon = eps[index]
                    plt.imshow(
                        self.generate_adversarial_image_pgd(
                            random_image[0], random_image[1], self.model
                        )
                    )
                    plt.axis("off")
            plt.tight_layout()
            plt.savefig(self.save_root + "/LAMP=" + self.lamp + "/CULTURE=" + i)
            plt.show()
            plt.close()
