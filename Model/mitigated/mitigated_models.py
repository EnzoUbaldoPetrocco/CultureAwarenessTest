#!/usr/bin/env python
__author__ = "Enzo Ubaldo Petrocco"
import sys

from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

sys.path.insert(1, "../")
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Flatten, Input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import Model
from keras.applications.resnet import ResNet50
from keras.applications.resnet_v2 import ResNet50V2
from keras.applications.efficientnet import EfficientNetB3
from keras.applications.efficientnet_v2 import EfficientNetV2S
from keras import layers, optimizers
from Model.GeneralModel import GeneralModelClass
import neural_structured_learning as nsl
import gc
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import math


class MitigatedModels(GeneralModelClass):
    def __init__(
        self,
        type="DL",
        culture=0,
        verbose_param=0,
        epochs=15,
        batch_size=1,
        learning_rate=1e-3,
        lambda_index=-1,
        n_cultures = 3
    ):
        """
        Initialization function for modeling mitigated ML models.
        We have narrowed the problems to image classification problems.
        :param type: selects the algorithm even if up to now "RESNET" is the only possible value.
        :param culture: selects the majority culture
        :param verbose_param: if enabled, the program logs more information
        :param learning_rate: hyperparameter for DL
        :param epochs: hyperparameter for DL
        :param batch_size: hyperparameter for DL
        :param lambda_index: select the gain of the regularizer in a logspace(-3, 2, 31)
        """
        GeneralModelClass.__init__(self, standard=0, n_cultures=n_cultures)
        self.type = type
        self.culture = culture
        self.verbose_param = verbose_param
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        if lambda_index >= 0:
            lambda_grid = np.logspace(-3, 2, 31)
            self.lamb = lambda_grid[lambda_index]
        else:
            self.lamb = 0

    def computeCIC(self, errs):
        tf.add(errs, -tf.math.reduce_min(errs))
        cic = tf.reduce_mean(errs)
        return cic

    def get_cic(self, valX, valY):
        losses = []
        valY = list(np.asarray(valY)[:, self.n_cultures])
        for out in range(self.n_cultures):
            yPred = self.model.predict(np.asarray(valX, dtype="int32"))
            yPred = list(np.asarray(yPred)[:, out])
            ls = tf.keras.losses.binary_crossentropy(valY, yPred)
            losses.append(ls)
        cic = float(self.computeCIC(losses))
        return cic

    def custom_loss(self):
        """
        This function implements the loss and the regularizer of the mitigation stratyegy
        :param out: related to the corresponding output to be optimized
        :return loss function
        """

        @tf.function
        def loss(y_true, y_pred):
            pred = tf.linalg.matmul(
                y_pred, y_true[:, 0 : self.n_cultures], transpose_b=True
            )
            pred = tf.linalg.tensor_diag_part(pred)
            ls = tf.keras.losses.binary_crossentropy(y_true[:, self.n_cultures], pred)
            return ls

        return loss

    def custom_accuracy(self):
        @tf.function
        def accuracy(y_true, y_pred):
            pred = tf.linalg.matmul(
                y_pred, y_true[:, 0 : self.n_cultures], transpose_b=True
            )
            pred = tf.linalg.tensor_diag_part(pred)

            acc = tf.keras.metrics.binary_accuracy(y_true[:, self.n_cultures], pred, threshold=0.5)
            return acc

        return accuracy

    @tf.function
    def regularizer(self, w):
        sum = tf.constant(0.0, dtype="float32")

        mean = tf.reduce_mean(w, axis=1)
        for i in range(self.n_cultures):
            sum += tf.math.square(tf.norm(w[:, i] - mean))

        res = (self.lamb) * sum
        return res

    def get_best_idx(self, losses: list, cics: list, tau=0.1):
        tmp_losses = losses.copy()
        n_ls = math.ceil(len(losses) * tau)

        pairs = []
        tmp_cics = []

        for i in range(n_ls):
            val = min(tmp_losses)
            idx = tmp_losses.index(val)
            pairs.append((val, idx))
            tmp_losses.remove(val)
            tmp_cics.append(cics[idx])

        mincic = min(tmp_cics)
        for i in range(n_ls):
            if mincic == cics[i]:
                idx = i

        return pairs[i][1]

    def ModelSelection(
        self,
        TS,
        VS,
        aug,
        show_imgs=False,
        batches=[32],
        lrs=[ 1e-5],
        fine_lrs=[1e-5, 1e-6],
        epochs=25,
        fine_epochs=10,
        nDropouts=[0.4],
        g=0.1,
    ):
        best_loss = np.inf
        losses = []
        cics = []

        lambdas = np.logspace(-3, 1, 1)
        for lmb in lambdas:
            self.lamb = lmb
            for b in batches:
                for lr in lrs:
                    for fine_lr in fine_lrs:
                        for nDropout in nDropouts:
                            with tf.device("/gpu:0"):
                                self.model = None
                                print(
                                    f"Training with: lamb={lmb}, batch_size={b}, lr={lr}, fine_lr={fine_lr}, nDropout={nDropout}"
                                )
                                history = self.DL(
                                    TS,
                                    VS,
                                    aug,
                                    show_imgs,
                                    b,
                                    lr,
                                    fine_lr,
                                    epochs,
                                    fine_epochs,
                                    nDropout,
                                    g=g,
                                )
                                loss = history.history["val_loss"][-1]
                                CIC = self.get_cic(VS[0], VS[1])
                                print(f"loss is {loss}, cic is {CIC}")
                                losses.append(loss)
                                cics.append(CIC)
                                if loss < best_loss:
                                    best_loss = loss
                                    best_bs = b
                                    best_lr = lr
                                    best_fine_lr = fine_lr
                                    best_nDropout = nDropout
                                self.model = None
                                gc.collect()

        idx = self.get_best_idx(losses, cics)
        best_loss = losses[idx]
        best_CIC = cics[idx]
        best_fine_lr_idx = idx % len(fine_lrs)
        best_fine_lr = fine_lrs[best_fine_lr_idx]
        best_lr_idx = math.floor(idx / len(fine_lrs)) % len(lrs)
        best_lr = lrs[best_lr_idx]
        best_bs_idx = math.floor(idx / (len(fine_lrs) * len(lrs))) % len(batches)
        best_bs = batches[best_bs_idx]
        best_lmb_idx = math.floor(idx / (len(fine_lrs) * len(batches) * len(lrs)))
        best_lmb = lambdas[best_lmb_idx]

        print(
            f"best_fine_lr_idx = {best_fine_lr_idx}, best_fine_lr = {best_fine_lr}, best_lr_idx = {best_lr_idx}, best_lr = {best_lr}, best_bs_idx = {best_bs_idx}, best_bs = {best_bs}, best_lmb_idx = {best_lmb}"
        )

        self.lamb = best_lmb
        with tf.device("/gpu:0"):
            print(
                f"Best loss:{best_loss}, best batch size:{best_bs}, best lr:{best_lr}, best fine_lr:{best_fine_lr}, best_dropout:{best_nDropout}, best lambda={best_lmb}, best CIC={best_CIC}"
            )
            TS = TS + VS
            self.DL(
                TS,
                None,
                aug,
                show_imgs,
                best_bs,
                best_lr,
                best_fine_lr,
                epochs,
                fine_epochs,
                best_nDropout,
                val=False,
                g=g,
            )

    def DL(
        self,
        TS,
        VS,
        aug=False,
        show_imgs=False,
        batch_size=32,
        lr=1e-3,
        fine_lr=1e-5,
        epochs=1,
        fine_epochs=1,
        nDropout=0.2,
        g=0.1,
        val=True,
    ):
        shape = np.shape(TS[0][0])
        n = np.shape(TS[0])

        if val:
            monitor_val = "val_loss"
        else:
            monitor_val = "loss"

        data_augmentation = keras.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(g / 10),
                layers.GaussianNoise(g),
                tf.keras.layers.RandomBrightness(g / 10),
                layers.RandomCrop(int(shape[0] * (1 - g)), int(shape[1] * (1 - g))),
                layers.RandomZoom(g / 5, g / 5),
                layers.Resizing(shape[0], shape[1]),
            ]
        )

        # Apply data augmentation to the training dataset
        train_datagen = ImageDataGenerator(
            preprocessing_function=lambda img: data_augmentation(img, training=aug)
        )
        X = tf.constant(TS[0], dtype="float32")
        y = tf.constant(TS[1], dtype="float32")
        train_generator = train_datagen.flow(x=X, y=y, batch_size=32)
        validation_generator = None
        if val:
            val_datagen = ImageDataGenerator()
            Xv = tf.constant(VS[0], dtype="float32")
            yv = tf.constant(VS[1], dtype="float32")
            validation_generator = val_datagen.flow(x=Xv, y=yv, batch_size=32)

        if show_imgs:
            # DISPLAY IMAGES
            # NOAUGMENTATION
            images = []
            for i in range(9):
                idx = np.random.randint(0, len(TS[0]) - 1)
                images.append((TS[0][idx], TS[1][idx]))
            plt.figure(figsize=(10, 10))
            for i, (image, label) in enumerate(images):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(image)
                plt.title(int(label))
                plt.axis("off")
            plt.show()

        # DIVIDE IN BATCHES
        if aug:
            if show_imgs:
                # DISPLAY IMAGES
                # AUGMENTATION
                idx = np.random.randint(0, len(TS) - 1)
                images = []
                images.append((TS[0][idx], TS[1][idx]))
                for ims, labels in images:
                    plt.figure(figsize=(10, 10))
                    for i in range(9):
                        ax = plt.subplot(3, 3, i + 1)

                        augmented_image = data_augmentation(
                            tf.expand_dims(ims, 0), training=True
                        )
                        plt.imshow(augmented_image[0].numpy().astype("int32"))
                        plt.title(int(labels))
                        plt.axis("off")
                    plt.show()

        # MODEL IMPLEMENTATION
        base_model = keras.applications.ResNet50V2(
            weights="imagenet",  # Load weights pre-trained on ImageNet.
            input_shape=shape,
            include_top=False,
        )  # Do not include the ImageNet classifier at the top.

        # Freeze the base_model
        base_model.trainable = False

        # Create  model on top
        inputs = keras.Input(shape=shape)
        # Pre-trained Xception weights requires that input be scaled
        # from (0, 255) to a range of (-1., +1.), the rescaling layer
        # outputs: `(inputs * scale) + offset`
        scale_layer = keras.layers.Rescaling(scale=1 / 255.0)
        if aug:
            x = data_augmentation(inputs)  # Apply random data augmentation
            x = scale_layer(x)
        else:
            x = scale_layer(inputs)

        # The base model contains batchnorm layers. We want to keep them in inference mode
        # when we unfreeze the base model for fine-tuning, so we make sure that the
        # base_model is running in inference mode here.
        x = base_model(x, training=False)
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dropout(nDropout)(x)  # Regularize with dropout
        outputs = keras.layers.Dense(
            self.n_cultures,
            # kernel_initializer="ones",
            kernel_regularizer=self.regularizer,
            activation='relu',
        )(x)
        # outputs = keras.layers.Dense(n_cultures)(x)
        self.model = keras.Model(inputs, outputs)

        lr_reduce = ReduceLROnPlateau(
            monitor=monitor_val,
            factor=0.1,
            patience=3,
            verbose=self.verbose_param,
            mode="max",
            min_lr=1e-9,
        )
        early = EarlyStopping(
            monitor=monitor_val,
            min_delta=0.001,
            patience=8,
            verbose=self.verbose_param,
            mode="auto",
        )
        callbacks = [early]

        # self.model.summary()
        # MODEL TRAINING
        self.model.compile(
            optimizer=keras.optimizers.Adam(lr),
            loss=self.custom_loss(),
            metrics=[self.custom_accuracy()],
            run_eagerly=True,
        )

        # ws = np.linalg.norm(self.model.layers[-1].weights)
        self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            verbose=self.verbose_param,
            callbacks=callbacks,
        )
        # ws2 = np.linalg.norm(self.model.layers[-1].weights)
        # print(f"Same = {ws2==ws}")

        # FINE TUNING
        base_model.trainable = True
        # self.model.summary()

        self.model.compile(
            optimizer=keras.optimizers.Adam(fine_lr),  # Low learning rate
            loss=self.custom_loss(),
            metrics=[self.custom_accuracy()],
        )

        history = self.model.fit(
            train_generator,
            epochs=fine_epochs,
            validation_data=validation_generator,
            verbose=self.verbose_param,
            callbacks=callbacks,
        )

        return history

    def fit(
        self,
        TS,
        VS=None,
        adversary=0,
        eps=0.05,
        mult=0.2,
        gradcam=False,
        out_dir="./",
        complete=0,
        aug=0,
        g=0.1,
    ):
        """
        General function for implementing model selection
        :param TS: training set
        :param VS: validation set
        :param adversary: if enabled, adversarial training is enabled
        :param eps: if adversary enabled, step size of adversarial training
        :param mult: if adversary enabled, multiplier of adversarial training
        :param gradcam: if enabled, gradcam callback is called
        :param out_dir: if gradcam enabled, output directory of gradcam heatmap
        :param complete: dummy argument
        """
        if self.type == "SVC":
            self.SVC(TS)
        elif self.type == "RFC":
            self.RFC(TS)
        elif self.type == "DL" or "RESNET":
            self.ModelSelection(TS, VS, aug=aug, g=g)
            """self.DL_model_selection(
                TS, VS, adversary, eps, mult, gradcam=gradcam, out_dir=out_dir
            )"""
        else:
            self.ModelSelection(TS, VS, aug=aug, g=g)

    def get_model_from_weights(self, size, adversary=0, eps=0.05, mult=0.2, path="./"):
        self.model = tf.keras.models.load_model(path)
