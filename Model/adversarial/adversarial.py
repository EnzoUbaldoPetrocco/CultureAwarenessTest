#!/usr/bin/env python
__author__ = "Enzo Ubaldo Petrocco"
import sys
import time

from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(1, "../")
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import layers
from Model.GeneralModel import GeneralModelClass
import gc
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
from datetime import datetime
#import keras_cv
from PIL import Image
from IPython.display import Image as IImage

random.seed(datetime.now().timestamp())
tf.random.set_seed(datetime.now().timestamp())


class AdversarialStandard(GeneralModelClass):
    def __init__(
        self,
        type="SVC",
        points=50,
        kernel="linear",
        verbose_param=0,
        learning_rate=1e-3,
        epochs=15,
        batch_size=1,
        weights=None,
        imbalanced=0,
        class_division=0,
        only_imb_imgs=0
    ):
        """
        Initialization function for modeling standard ML models.
        We have narrowed the problems to image classification problems.
        I have implemented SVM (with linear and gaussian kernel) and Random Forest, using scikit-learn library;
        ResNet using Tensorflow library.
        :param type: selects the algorithm "SVC", "RFC" and "RESNET" are possible values.
        :param points: n of points in gridsearch for SVC and RFC
        :param kernel: type of kernel for SVC: "linear" and "gaussian" are possible values.
        :param verbose_param: if enabled, the program logs more information
        :param learning_rate: hyperparameter for DL
        :param epochs: hyperparameter for DL
        :param batch_size: hyperparameter for DL
        """
        GeneralModelClass.__init__(
            self, standard=1, adversarial=1, imbalanced=imbalanced
        )
        self.type = type
        self.points = points
        self.kernel = kernel
        self.verbose_param = verbose_param
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weights = np.ones(self.n_cultures)
        self.class_division = class_division
        self.only_imb_imgs=only_imb_imgs
        if weights is not None:
            self.weights = weights

    def SVC(self, TS):
        """
        This function performs the model selection on SVM for Classification
        :param TS: union between training and validation set
        :return the best model
        """
        if self.kernel == "rbf":
            logspaceC = np.logspace(-4, 3, self.points)  # np.logspace(-2,2,self.points)
            logspaceGamma = np.logspace(
                -4, 3, self.points
            )  # np.logspace(-2,2,self.points)
            grid = {"C": logspaceC, "kernel": [self.kernel], "gamma": logspaceGamma}
        if self.kernel == "linear":
            logspaceC = np.logspace(-4, 3, self.points)  # np.logspace(-2,2,self.points)
            logspaceGamma = np.logspace(
                -4, 3, self.points
            )  # np.logspace(-2,2,self.points)
            grid = {"C": logspaceC, "kernel": [self.kernel]}

        MS = GridSearchCV(
            estimator=SVC(),
            param_grid=grid,
            scoring="balanced_accuracy",
            cv=10,
            verbose=self.verbose_param,
        )
        # training set is divided into (X,y)
        TS = np.array(TS, dtype=object)
        del TS
        X = list(TS[:, 0])
        y = list(TS[:, 1])
        print("SVC TRAINING")
        H = MS.fit(X, y)
        # Check that C and gamma are not the extreme values
        print(f"C best param {H.best_params_['C']}")
        # print(f"gamma best param {H.best_params_['gamma']}")
        self.model = H

    def RFC(self, TS):
        """
        This function performs the model selection on Random Forest for Classification
        :param TS: union between training and validation set
        :return the best model
        """
        rfc = RandomForestClassifier(random_state=42)
        logspace_max_depth = []
        for i in np.logspace(0, 3, self.points):
            logspace_max_depth.append(int(i))
        param_grid = {
            "n_estimators": [500],  # logspace_n_estimators,
            "max_depth": logspace_max_depth,
        }

        CV_rfc = GridSearchCV(
            estimator=rfc, param_grid=param_grid, cv=5, verbose=self.verbose_param
        )
        # training set is divided into (X,y)
        TS = np.array(TS, dtype=object)
        X = list(TS[:, 0])
        y = list(TS[:, 1])
        del TS
        print("RFC TRAINING")
        H = CV_rfc.fit(X, y)
        # print(CV_rfc.best_params_)
        self.model = H

    @tf.function
    def generate_adversarial_image(self, img, lbl, model, epsilon=0.1, aug=False):
        img = tf.expand_dims(img, axis=0)
        lbl = tf.expand_dims(lbl, axis=0)
        img = tf.convert_to_tensor(img)
        lbl = tf.convert_to_tensor(lbl)
        
        with tf.GradientTape() as tape:
            tape.watch(img)
            prediction = model(img, training=False)
            loss = tf.keras.losses.categorical_crossentropy(lbl, prediction)
        gradient = tape.gradient(loss, img)
        signed_grad = tf.sign(gradient)
        img = img / 255.0
        sum = tf.cast(epsilon, dtype=np.float32) * tf.cast(
            signed_grad, dtype=np.float32
        )
        adversarial_img = tf.cast(img, dtype=np.float32) + sum
        adversarial_img = adversarial_img * 255.0
        return tf.clip_by_value(adversarial_img, 0, 255)
    
    @tf.function
    def generate_adversarial_image_pgd(self, img, lbl, model,  epsilon=0.1, alpha=0.01, num_iter=20):
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
        
        for i in range(num_iter):
            with tf.GradientTape() as tape:
                tape.watch(x_adv)
                prediction = model(x_adv)
                loss = tf.keras.losses.sparse_categorical_crossentropy(lbl, prediction)
            
            # Get the gradients of the loss w.r.t. the input image.
            gradients = tape.gradient(loss, x_adv)
            
            # Perform the gradient ascent step
            perturbations = alpha * tf.sign(gradients)
            x_adv = x_adv + perturbations
            
            # Project the perturbation onto the epsilon ball
            x_adv = tf.clip_by_value(x_adv, img - epsilon, img + epsilon)
            x_adv = tf.clip_by_value(x_adv, 0, 1)  # Ensure the pixel values are still valid
            
        return x_adv

        with tf.GradientTape() as tape:
            tape.watch(img)
            prediction = model(img, training=False)
            loss = tf.keras.losses.categorical_crossentropy(lbl, prediction)
        gradient = tape.gradient(loss, img)
        signed_grad = tf.sign(gradient)
        img = img / 255.0
        sum = tf.cast(epsilon, dtype=np.float32) * tf.cast(
            signed_grad, dtype=np.float32
        )
        adversarial_img = tf.cast(img, dtype=np.float32) + sum
        adversarial_img = adversarial_img * 255.0
        return tf.clip_by_value(adversarial_img, 0, 255)

    def generate_adversarial_samples(self, adv_train_generator, model, epsilon=0.1, aug=False):
        adversarial_images = []
        for img, lbl in adv_train_generator:
            adversarial_img = self.generate_adversarial_image(img, lbl, model, epsilon, aug)
            adversarial_images.append(adversarial_img)
        return tf.convert_to_tensor(adversarial_images)

    def remove_data_aug(self, model :keras.Model):  
        input = keras.Input(shape=self.shape)
        x = model.layers[2](input)
        x = model.layers[3](x)
        x = model.layers[4](x)
        x = model.layers[5](x)
        y = model.layers[6](x)
        truncated_model = keras.Model(inputs = input, outputs = y)   
        truncated_model.summary()
        return truncated_model

    def generate_adv_images_using_text(self, TS):
        keras.mixed_precision.set_global_policy("mixed_float16")

        model = keras_cv.models.StableDiffusionV2(jit_compile=True)   
        prompt_1 = "An image of a Chinese lamp"
        prompt_2 = "An image of a French lamp"
        interpolation_steps = 5

        encoding_1 = tf.squeeze(model.encode_text(prompt_1))
        encoding_2 = tf.squeeze(model.encode_text(prompt_2))

        interpolated_encodings = tf.linspace(encoding_1, encoding_2, interpolation_steps)
        print(f"Encoding shape: {encoding_1.shape}")
        
        noise = tf.random.normal((512 // 8, 512 // 8, 4), seed=random.seed(datetime.now().timestamp()))
        print(f"Before")
        images = model.generate_image(
            interpolated_encodings,
            batch_size=interpolation_steps,
            diffusion_noise=noise)
        print(f"After")
        
        def export_as_gif(filename, images, frames_per_second=10, rubber_band=False):
            if rubber_band:
                images += images[2:-1][::-1]
            images[0].save(
                filename,
                save_all=True,
                append_images=images[1:],
                duration=1000 // frames_per_second,
                loop=0)

        export_as_gif(
            "./doggo-and-fruit-6.gif",
            [Image.fromarray(img) for img in images],
            frames_per_second=2,
            rubber_band=True)
              
        IImage("./doggo-and-fruit-6.gif")
    
    def LearningAdversarially(
        self,
        TS,
        VS,
        aug,
        show_imgs=True,
        batches=[32],
        lrs=[1e-2, 1e-3, 1e-4, 1e-5],
        fine_lrs=[1e-5],
        epochs=30,
        fine_epochs=10,
        nDropouts=[0.4],
        g=0.1,
        save=False,
        path="./",
        eps=0.1,
        text_adv=0,
    ):
        class_division = self.class_division
        if text_adv:
            self.generate_adv_images_using_text(TS)
        
        if not self.only_imb_imgs:
            if self.imbalanced:
                TS = self.ImbalancedTransformation(TS)
                VS = self.ImbalancedTransformation(VS) 
        else:
            TS2 = self.ImbalancedTransformation(TS)
            VS2 = self.ImbalancedTransformation(VS2)

        epsilons = np.logspace(-3, 0, 5)
        images = []
        for i in range(4):
            idx = np.random.randint(0, len(TS[0]))
            images.append((TS[0][idx], TS[1][idx]))

        if class_division:
            adversarial_model = []
            print(f"ADVERSARIAL USING CLASS DIVISION")
            for j in range(2):
                tempX = [
                    TS[0][i]
                    for i in range(len(TS[0]))
                    if TS[1][i][self.n_cultures] == j
                ]
                tempY = [
                    TS[1][i]
                    for i in range(len(TS[1]))
                    if TS[1][i][self.n_cultures] == j
                ]
                tempTS = (tempX, tempY)
                tempX = [
                    VS[0][i]
                    for i in range(len(VS[0]))
                    if VS[1][i][self.n_cultures] == j
                ]
                tempY = [
                    VS[1][i]
                    for i in range(len(VS[1]))
                    if VS[1][i][self.n_cultures] == j
                ]
                tempVS = (tempX, tempY)
                self.ModelSelection(
                    TS=tempTS,
                    VS=tempVS,
                    aug=aug,
                    show_imgs=show_imgs,
                    batches=batches,
                    lrs=lrs,
                    fine_lrs=fine_lrs,
                    epochs=epochs,
                    fine_epochs=fine_epochs,
                    nDropouts=nDropouts,
                    g=g,
                    save=save,
                    path=path,
                    adv=1,
                    adversarial_model=None,
                    eps=eps,
                )
                if aug:
                    adversarial_model.append(self.remove_data_aug(self.model))
                else:
                    adversarial_model.append(self.model)
                self.model = None

            (imgs, ys) = TS[0], TS[1]
            for i in range(len(imgs)):
                img = imgs[i]
                y = ys[i]
                imgs[i] = self.generate_adversarial_image(tf.cast(img, dtype=np.float32), tf.cast(y[0:self.n_cultures], dtype=np.float32), adversarial_model[int(y[self.n_cultures])], eps, aug)[0]
            
            (imgs, ys) = VS[0], VS[1]
            for i in range(len(imgs)):
                img = imgs[i]
                y = ys[i]
                imgs[i] = self.generate_adversarial_image(tf.cast(img, dtype=np.float32), tf.cast(y[0:self.n_cultures], dtype=np.float32), adversarial_model[int(y[self.n_cultures])], eps, aug)[0]

            if show_imgs:
                for ep in epsilons:
                    plt.figure(figsize=(10, 10))
                    c = 1
                    for i, (image, label) in enumerate(images):
                        ax = plt.subplot(4, 2, c)
                        plt.imshow(image)
                        plt.title(label)
                        ax = plt.subplot(4, 2, c + 1)
                        c = c + 2
                        adv_image = self.generate_adversarial_image(
                            image * 1.0,
                            label[0 : self.n_cultures],
                            adversarial_model[int(label[self.n_cultures])],
                            epsilon=ep,
                            aug=aug
                        )[0]
                        plt.imshow(adv_image / 255.0)
                        plt.title(label)
                        plt.axis("off")
                    plt.show()

        else:
            self.ModelSelection(
                TS=TS,
                VS=VS,
                aug=aug,
                show_imgs=show_imgs,
                batches=batches,
                lrs=lrs,
                fine_lrs=fine_lrs,
                epochs=epochs,
                fine_epochs=fine_epochs,
                nDropouts=nDropouts,
                g=g,
                save=save,
                path=path,
                adv=1,
                adversarial_model=None,
                eps=eps,
                class_division=0,
            )
            if aug:
                adversarial_model = self.remove_data_aug(self.model)
            else:
                adversarial_model = self.model
            adversarial_model.summary()

            (imgs, ys) = TS[0], TS[1]
            for i in range(len(imgs)):
                img = imgs[i]
                y = ys[i]
                imgs[i] = self.generate_adversarial_image(tf.cast(img, dtype=np.float32), tf.cast(y[0:self.n_cultures], dtype=np.float32), adversarial_model, eps, aug)[0]
                
            
            (imgs, ys) = VS[0], VS[1]
            for i in range(len(imgs)):
                img = imgs[i]
                y = ys[i]
                imgs[i] = self.generate_adversarial_image(tf.cast(img, dtype=np.float32), tf.cast(y[0:self.n_cultures], dtype=np.float32), adversarial_model, eps, aug)[0]
                
            ###############################
            ####### SHOW DIFFERENT IMAGES BASED ON EPS #########
            if show_imgs:
                for ep in epsilons:
                    plt.figure(figsize=(10, 10))
                    c = 1
                    for i, (image, label) in enumerate(images):
                        ax = plt.subplot(4, 2, c)
                        plt.imshow(image)
                        plt.title(label)
                        ax = plt.subplot(4, 2, c + 1)
                        c = c + 2
                        adv_image = self.generate_adversarial_image(
                            image * 1.0,
                            label[0 : self.n_cultures],
                            adversarial_model,
                            epsilon=ep,
                            aug=aug
                        )[0]
                        plt.imshow(adv_image / 255.0)
                        plt.title(label)
                        plt.axis("off")
                    plt.show()

        self.model = None
        self.ModelSelection(
            TS=TS,
            VS=VS,
            aug=aug,
            show_imgs=show_imgs,
            batches=batches,
            lrs=lrs,
            fine_lrs=fine_lrs,
            epochs=epochs,
            fine_epochs=fine_epochs,
            nDropouts=nDropouts,
            g=g,
            save=save,
            path=path,
            adv=0,
            adversarial_model=adversarial_model,
            eps=eps,
            class_division=class_division,
        )

    def ModelSelection(
        self,
        TS,
        VS,
        aug,
        show_imgs=False,
        batches=[32],
        lrs=[1e-2, 1e-3, 1e-4, 1e-5],
        fine_lrs=[1e-5],
        epochs=30,
        fine_epochs=10,
        nDropouts=[0.4],
        g=0.1,
        save=False,
        path="./",
        adv=0,  # if 1-> adversarial model is trained, if 0 -> actual model is used
        adversarial_model=None,
        eps=0.1,
        class_division=0,
    ):

        if self.verbose_param:
            tf.get_logger().setLevel(4)
        best_loss = np.inf
        for b in batches:
            for lr in lrs:
                for fine_lr in fine_lrs:
                    for nDropout in nDropouts:
                        self.model = None
                        gc.collect()
                        tf.get_logger().info(
                            f"Training with: batch_size={b}, lr={lr}, fine_lr={fine_lr}, nDropout={nDropout}"
                        )
                        sys.stdout.write("\r")
                        loss = self.DL(
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
                            adv=adv,
                            adversarial_model=adversarial_model,
                            eps=eps,
                            class_division=class_division,
                        )

                        if loss < best_loss:
                            best_loss = loss
                            best_bs = b
                            best_lr = lr
                            best_fine_lr = fine_lr
                            best_nDropout = nDropout

                        self.model = None
                        gc.collect()

        print(
            f"Best loss:{best_loss}, best batch size:{best_bs}, best lr:{best_lr}, best fine_lr:{best_fine_lr}, best_dropout:{best_nDropout}"
        )
        list(TS).append(VS)
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
            adv=adv,
            adversarial_model=adversarial_model,
            eps=eps,
            class_division=class_division,
        )

        if save:
            self.save(path)

    def ImbalancedTransformation(self, TS):
        newX = []
        newY = []
        X = TS[0]
        Y = TS[1]
        if not self.only_imb_imgs:
            for i in range(len(X)):
                img = X[i]
                label = Y[i]
                for j in range(
                    int(1 / self.weights[label.index(1.0)])
                ):  # I use the inverse of the total proportion for augmenting the dataset
                    newX.append(np.asarray(img))  
                    newY.append(np.asarray(label))
        else:
            for i in range(len(X)):
                img = X[i]
                label = Y[i]
                if int(1 / self.weights[label.index(1.0)])>1:
                    for j in range(
                        int(1 / self.weights[label.index(1.0)])
                    ):  # I use the inverse of the total proportion for augmenting the dataset
                        newX.append(np.asarray(img))  
                        newY.append(np.asarray(label))
        del TS
        return (newX, newY)

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
        adv=0,  # if 1-> adversarial model is trained, if 0 -> actual model is used
        adversarial_model=None,
        eps=0.1,
        class_division=0,
    ):
        with tf.device("/gpu:0"):
            shape = np.shape(TS[0][0])
            self.shape = shape

            if val:
                monitor_val = "val_loss"
            else:
                monitor_val = "loss"

            data_augmentation = keras.Sequential(
                [
                    layers.RandomFlip("horizontal"),
                    layers.RandomRotation(0.01),
                    layers.GaussianNoise(g),
                    tf.keras.layers.RandomBrightness(0.01),
                    layers.RandomZoom(g, g),
                    layers.Resizing(shape[0], shape[1]),
                ]
            )

            if show_imgs and (not adv):
                # DISPLAY IMAGES
                # NOAUGMENTATION
                images = []
                for i in range(4):
                    idx = np.random.randint(0, len(TS[0]) - 1)
                    images.append((TS[0][idx], TS[1][idx]))
                plt.figure(figsize=(10, 10))
                for i, (image, label) in enumerate(images):
                    ax = plt.subplot(4, 2, i + 1)
                    plt.imshow(image)
                    plt.title(label)
                    ax = plt.subplot(4, 2, i + 5)
                    if class_division:
                        adv_image = self.generate_adversarial_image(
                            data_augmentation(image, training=aug) * 1.0,
                            label[0 : self.n_cultures],
                            adversarial_model[int(label[self.n_cultures])],
                            epsilon=eps,
                            aug=aug
                        )[0]
                    else:
                        adv_image = self.generate_adversarial_image(
                            data_augmentation(image, training=aug) * 1.0,
                            label[0 : self.n_cultures],
                            adversarial_model,
                            epsilon=eps,
                            aug=aug
                        )[0]
                    plt.imshow(adv_image / 255.0)
                    plt.title(label)
                    plt.axis("off")
                plt.show()

            validation_generator = None
            train_generator = tf.data.Dataset.from_tensor_slices(TS)
            # train_generator = tf.random.shuffle(int(train_generator.cardinality()/batch_size))

            if adv:  # adversarial model
                train_generator = train_generator.map(
                    lambda img, y: (
                        data_augmentation(img, training=aug),
                        y[0 : self.n_cultures],
                    )
                )
            else:  # actual model
                train_generator = train_generator.map(
                    lambda img, y: (
                        data_augmentation(img, training=aug),
                        y[self.n_cultures],
                    )
                )

            train_generator = (
                train_generator.cache().batch(batch_size).prefetch(buffer_size=10)
            )
            if val:
                validation_generator = tf.data.Dataset.from_tensor_slices(VS)

                if adv:
                    validation_generator = validation_generator.map(
                        lambda img, y: (
                            data_augmentation(img, training=aug),
                            y[0 : self.n_cultures],
                        )
                    )
                else:
                    validation_generator = validation_generator.map(
                        lambda img, y: (
                            data_augmentation(img, training=aug),
                            y[self.n_cultures],
                        )
                    )
                validation_generator = (
                    validation_generator.cache()
                    .batch(batch_size)
                    .prefetch(buffer_size=10)
                )

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

            adversarial_model = None

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
            scale_layer = keras.layers.Rescaling(scale=1 / 255.0)
            if aug:
                x = data_augmentation(inputs)  # Apply random data augmentation
                x = scale_layer(x)
            else:
                x = scale_layer(inputs)

            x = base_model(x, training=False)
            x = keras.layers.GlobalAveragePooling2D()(x)
            x = keras.layers.Dropout(nDropout)(x)  # Regularize with dropout
            if adv:
                outputs = keras.layers.Dense(3, activation="softmax")(x)
            else:
                outputs = keras.layers.Dense(1, activation="sigmoid")(x)
            self.model = keras.Model(inputs, outputs)

            if adv:
                bcemetric = keras.losses.CategoricalCrossentropy(from_logits=True)
                train_acc_metric = keras.metrics.CategoricalAccuracy()
            else:
                bcemetric = keras.losses.BinaryCrossentropy(from_logits=True)
                train_acc_metric = keras.metrics.BinaryAccuracy()

            lr_reduce = ReduceLROnPlateau(
                monitor=monitor_val,
                factor=0.2,
                patience=5,
                verbose=self.verbose_param,
                mode="max",
                min_lr=1e-9,
            )
            early = EarlyStopping(
                monitor=monitor_val,
                min_delta=0.001,
                patience=10,
                verbose=self.verbose_param,
                mode="auto",
            )
            callbacks = [early, lr_reduce]

            # self.model.summary()

            # MODEL TRAINING
            self.model.compile(
                optimizer=keras.optimizers.Adam(lr),
                loss=bcemetric,
                metrics=[train_acc_metric],
                # run_eagerly=True
            )

            self.model.fit(
                train_generator,
                epochs=epochs,
                validation_data=validation_generator,
                verbose=self.verbose_param,
                callbacks=callbacks,
            )

            # FINE TUNING
            base_model.trainable = True
            # self.model.summary()

            self.model.compile(
                optimizer=keras.optimizers.Adam(fine_lr),
                loss=bcemetric,
                metrics=[train_acc_metric],
                # run_eagerly=True
            )

            history = self.model.fit(
                train_generator,
                epochs=fine_epochs,
                validation_data=validation_generator,
                verbose=self.verbose_param,
                callbacks=callbacks,
            )
            tf.keras.backend.clear_session()
            return history.history[monitor_val][-1]

    # @tf.function
    def train_loop(
        self,
        model,
        epochs,
        train_dataset,
        val_dataset,
        loss_fn,
        optimizer: keras.optimizers.Adam,
        batch_size,
        train_acc_metric: keras.metrics.BinaryAccuracy,
        val_acc_metric: keras.metrics.BinaryAccuracy,
        bcemetric: keras.metrics.BinaryCrossentropy,
        monitor_val,
        val,
        val_bcemetric: keras.metrics.BinaryCrossentropy = None,
        n=0,
        adv=0,
        adversarial_model=None,
        eps=0,
    ):

        @tf.function
        def create_adversarial_pattern(model, input_image, input_label):
            with tf.device("/gpu:0"):
                with tf.GradientTape() as tape:
                    tape.watch(input_image)
                    prediction = model(input_image)
                    loss = tf.keras.losses.categorical_crossentropy(
                        input_label, prediction
                    )

                gradient = tape.gradient(loss, input_image)
                signed_grad = tf.sign(gradient)
                return signed_grad

        # Create adversarial samples
        @tf.function
        def generate_adversarial_samples(model, images, labels, shape, epsilon=0.1):
            with tf.device("/gpu:0"):
                adversarial_images = []
                for img, lbl in zip(images, labels):
                    img = tf.convert_to_tensor(img.reshape((1, shape[0], shape[1], 3)))
                    lbl = tf.convert_to_tensor(lbl.reshape((1, self.n_cultures)))
                    perturbations = create_adversarial_pattern(model, img, lbl)
                    adversarial_img = img + epsilon * perturbations
                    adversarial_img = tf.clip_by_value(adversarial_img, 0, 1)
                    adversarial_images.append(adversarial_img.numpy())
                return tf.convert_to_tensor(adversarial_images)

        @tf.function
        def train_step(x, y):
            with tf.GradientTape() as tape:
                logits = model(x, training=True)
                loss_value = loss_fn(y, logits)
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            train_acc_metric.update_state(y, logits)
            bcemetric.update_state(y, logits)

        @tf.function
        def test_step(x, y):
            val_logits = model(x, training=False)
            val_acc_metric.update_state(y, val_logits)
            val_bcemetric.update_state(y, val_logits)

        # implement reduce learning rate on pleateu
        rlrop = {
            "factor": 0.2,
            "patience": 5,
            "min_lr": 1e-9,
            "max_val": np.inf,
            "prec_step": None,
        }

        # implemet early_stopping
        es = {
            "min_delta": 0.001,
            "patience": 10,
            "max_val": np.inf,
            "prec_step": None,
        }

        # print(f"model.trainable_weights are {model.trainable_weights}")
        values = {
            "loss": tf.constant(0.0, dtype=float),
            "val_loss": tf.constant(0.0, dtype=float),
        }
        for epoch in range(epochs):

            sys.stdout.write("\r")
            tf.get_logger().info(f"Epoch: {epoch}")
            pbt = tf.keras.utils.Progbar(n)
            start_time = time.time()
            values["loss"] = tf.constant(0.0, dtype=float)
            # Iterate over the batches of the dataset.
            step = 0
            for x_batch_train, y_batch_train in train_dataset:
                if adv:
                    x_batch_train = generate_adversarial_samples(
                        adversarial_model,
                        x_batch_train,
                        y_batch_train,
                        x_batch_train.shape,
                        epsilon=eps,
                    )

                train_step(x_batch_train, y_batch_train)
                # print(f"loss value is {loss_value}")

                pbt.add(
                    x_batch_train.shape[0],
                    values=[
                        ("loss", bcemetric.result()),
                        ("acc", train_acc_metric.result()),
                    ],
                )
                step += 1

            # Display metrics at the end of each epoch.
            train_acc = train_acc_metric.result()
            # Reset training metrics at the end of each epoch
            train_acc_metric.reset_states()

            loss = bcemetric.result()
            values["loss"] = loss
            bcemetric.reset_states()

            values["val_loss"] = tf.constant(0.0, dtype=float)
            # Run a validation loop at the end of each epoch.
            if val:
                for x_batch_val, y_batch_val in val_dataset:
                    test_step(x_batch_val, y_batch_val)

                val_acc = val_acc_metric.result()
                val_acc_metric.reset_states()

                val_loss = val_bcemetric.result()
                values["val_loss"] = val_loss
                val_bcemetric.reset_states()

                # tf.get_logger().info("Validation acc: %.4f" , float(val_acc))
                pbt.add(
                    n - x_batch_train.shape[0] * step + 1,
                    values=[
                        ("Time", time.time() - start_time),
                        ("loss", val_bcemetric.result()),
                        ("acc", train_acc),
                        ("val_loss", values["val_loss"]),
                        ("val_acc", val_acc),
                        ("lr", model.optimizer.lr),
                    ],
                )
            # tf.get_logger().info("Time taken: %.2fs" , time.time() - start_time)
            else:
                pbt.add(
                    n - x_batch_train.shape[0] * step + 1,
                    values=[
                        ("Time", time.time() - start_time),
                        ("loss", val_bcemetric.result()),
                        ("acc", train_acc),
                        ("lr", model.optimizer.lr),
                    ],
                )
            sys.stdout.write("\r")

            train_dataset = train_dataset.shuffle(batch_size)
            if val:
                val_dataset = val_dataset.shuffle(batch_size)

            # At the end of the epoch I have to call my callbacks
            if rlrop["max_val"] < values[monitor_val]:
                rlrop["prec_step"] += 1
            else:
                rlrop["prec_step"] = 0
                rlrop["max_val"] = values[monitor_val]

            if es["max_val"] < values[monitor_val]:
                es["prec_step"] += 1
            else:
                es["prec_step"] = 0
                es["max_val"] = values[monitor_val]

            if rlrop["prec_step"] > rlrop["patience"]:
                newlr = max(rlrop["factor"] * model.optimizer.lr, rlrop["min_lr"])
                tf.keras.backend.set_value(model.optimizer.lr, newlr)
                sys.stdout.write("\r")
                tf.get_logger().info(f"Reducing Learning rate to: {newlr:.9f}")
                sys.stdout.write("\r")

            if es["prec_step"] > es["patience"]:
                sys.stdout.write("\r")
                tf.get_logger().info(f"Early stopping")
                sys.stdout.write("\r")
                break

        return values["loss"], values["val_loss"]

    def fit(
        self,
        TS,
        VS=None,
        aug=0,
        g=0.1,
        eps=0.3,
        mult=0.2,
        gradcam=0,
        out_dir="./",
        complete=0,
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
            self.LearningAdversarially(TS, VS, aug=aug, g=g, path=out_dir, eps=eps)
            """self.DL_model_selection(
                TS, VS, adversary, eps, mult, gradcam=gradcam, out_dir=out_dir
            )"""
        else:
            self.LearningAdversarially(TS, VS, aug=aug, g=g, path=out_dir, eps=eps)
