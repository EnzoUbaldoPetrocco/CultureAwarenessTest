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


class AdversarialProcessing(keras.models.Model):

    def __init__(self, epsilon):
        super(AdversarialProcessing, self).__init__()
        self.epsilon = epsilon

    def __call__(self, batch, logs=None):
        adversarial_images = []
        for img, lbl in batch:
            img = tf.convert_to_tensor(
                img.reshape((1, self.input_shape[0], self.input_shape[1], 3))
            )
            lbl = tf.convert_to_tensor(lbl.reshape((1, 1)))
            with tf.GradientTape() as tape:
                tape.watch(img)
                prediction = self.model(img)
                loss = tf.keras.losses.binary_crossentropy(lbl, prediction)
            gradient = tape.gradient(loss, img)
            signed_grad = tf.sign(gradient)
            adversarial_img = img + self.epsilon * signed_grad
            adversarial_img = tf.clip_by_value(adversarial_img, 0, 1)
            adversarial_images.append(adversarial_img)
        return tf.convert_to_tensor(adversarial_images)

    # Create adversarial samples
    def generate_adversarial_samples(self, images, labels, epsilon=0.1):
        with tf.device("/gpu:0"):
            adversarial_images = []
            for img, lbl in zip(images, labels):
                img = tf.convert_to_tensor(
                    img.reshape((1, self.input_shape[0], self.input_shape[1], 3))
                )
                lbl = tf.convert_to_tensor(lbl.reshape((1, 1)))
                with tf.GradientTape() as tape:
                    tape.watch(img)
                    prediction = self.model(img)
                    loss = tf.keras.losses.categorical_crossentropy(lbl, prediction)
                gradient = tape.gradient(loss, img)
                signed_grad = tf.sign(gradient)
                adversarial_img = img + epsilon * signed_grad
                adversarial_img = tf.clip_by_value(adversarial_img, 0, 1)
                adversarial_images.append(adversarial_img)
            return tf.convert_to_tensor(adversarial_images)


#!/usr/bin/env python
__author__ = "Enzo Ubaldo Petrocco"
import sys

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
        GeneralModelClass.__init__(self, standard=1)
        self.type = type
        self.points = points
        self.kernel = kernel
        self.verbose_param = verbose_param
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

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

    def create_adversarial_pattern(self, model, input_image, input_label):
        with tf.device("/gpu:0"):
            with tf.GradientTape() as tape:
                tape.watch(input_image)
                prediction = model(input_image)
                loss = tf.keras.losses.categorical_crossentropy(input_label, prediction)

            gradient = tape.gradient(loss, input_image)
            signed_grad = tf.sign(gradient)
            return signed_grad

    # Create adversarial samples
    def generate_adversarial_samples(self, model, images, labels, shape, epsilon=0.1):
        with tf.device("/gpu:0"):
            adversarial_images = []
            for img, lbl in zip(images, labels):
                img = tf.convert_to_tensor(img.reshape((1, shape[0], shape[1], 3)))
                lbl = tf.convert_to_tensor(lbl.reshape((1, 1)))
                perturbations = self.create_adversarial_pattern(model, img, lbl)
                adversarial_img = img + epsilon * perturbations
                adversarial_img = tf.clip_by_value(adversarial_img, 0, 1)
                adversarial_images.append(adversarial_img.numpy())
            return tf.convert_to_tensor(adversarial_images)

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
    ):
        if self.verbose_param:
            tf.get_logger().setLevel(0)
        best_loss = np.inf
        for b in batches:
            for lr in lrs:
                for fine_lr in fine_lrs:
                    for nDropout in nDropouts:

                        self.model = None
                        gc.collect()
                        print(
                            f"Training with: batch_size={b}, lr={lr}, fine_lr={fine_lr}, nDropout={nDropout}"
                        )
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

        if save:
            self.save(path)

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
        with tf.device("/gpu:0"):
            # TEMP 
            TS=np.asarray(TS)
            VS=np.asarray(VS)

            TS[1]=np.asarray(TS[1])
            VS[1]=np.asarray(VS[1])

            print(f"Shape of TS[1]: {np.shape(TS[1])}")
            print(f"Shape of TS[1][]:,0]: {np.shape(TS[1][:,0])}")
            print(f"Shape of VS[1]: {np.shape(VS[1])}")

            TS[1]=np.asarray(TS[1])[:,0]
            VS[1]=np.asarray(VS[1])[:,0]

            print(f"Shape of TS[1]: {np.shape(TS[1])}")
            print(f"Shape of VS[1]: {np.shape(VS[1])}")

            TS = tuple(TS)
            VS=tuple(VS)
            #MANNAGGHIA


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

            validation_generator = None
            train_generator = tf.data.Dataset.from_tensor_slices(TS)
            if aug:
                train_ds = train_ds.map(lambda img, y: (data_augmentation(img, training=aug), y))

            train_generator = train_generator.cache().batch(batch_size).prefetch(buffer_size=10)
            if val:
                validation_generator = tf.data.Dataset.from_tensor_slices(VS)
                if aug:
                    validation_ds = validation_ds.map(lambda img, y: (data_augmentation(img, training=aug), y))
                validation_generator = validation_generator.cache().batch(batch_size).prefetch(buffer_size=10)    

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
            scale_layer = keras.layers.Rescaling(scale=1 / 255.0)
            if aug:
                x = data_augmentation(inputs)  # Apply random data augmentation
                x = scale_layer(x)
            else:
                x = scale_layer(inputs)

            x = base_model(x, training=False)
            x = keras.layers.GlobalAveragePooling2D()(x)
            x = keras.layers.Dropout(nDropout)(x)  # Regularize with dropout
            outputs = keras.layers.Dense(1, activation="sigmoid")(x)
            self.model = keras.Model(inputs, outputs)

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
                loss=keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=[keras.metrics.BinaryAccuracy()],
            )


            loss, val_loss = self.train_loop(
                epochs=epochs,
                train_dataset=train_generator,
                val_dataset=validation_generator,
                loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
                optimizer=keras.optimizers.Adam(lr),
                batch_size=batch_size,
                train_acc_metric=keras.metrics.BinaryAccuracy(),
                val_acc_metric=keras.metrics.BinaryAccuracy(),
                monitor_val=monitor_val,
                val=val
            )

            # FINE TUNING
            base_model.trainable = True
            # self.model.summary()

            self.model.compile(
                optimizer=keras.optimizers.Adam(fine_lr),  # Low learning rate
                loss=keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=[keras.metrics.BinaryAccuracy()],
            )

            loss, val_loss = self.train_loop(
                epochs=epochs,
                train_dataset=train_generator,
                val_dataset=validation_generator,
                loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
                optimizer=keras.optimizers.Adam(fine_lr),
                batch_size=batch_size,
                train_acc_metric=keras.metrics.BinaryAccuracy(),
                val_acc_metric=keras.metrics.BinaryAccuracy(),
                monitor_val=monitor_val,
                val=val
            )
            
            tf.keras.backend.clear_session()
            if val:
                return val_loss
            else:
                return loss
            

    def train_loop(
        self,
        epochs,
        train_dataset,
        val_dataset,
        loss_fn,
        optimizer,
        batch_size,
        train_acc_metric,
        val_acc_metric,
        monitor_val,
        val
    ):
        @tf.function
        def train_step(x, y):
            with tf.GradientTape() as tape:
                logits = self.model(x, training=True)
                loss_value = loss_fn(y, logits)
            grads = tape.gradient(loss_value, self.model.trainable_weights)
            optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
            train_acc_metric.update_state(y, logits)
            return loss_value

        @tf.function
        def test_step(x, y):
            val_logits = self.model(x, training=False)
            val_acc_metric.update_state(y, val_logits)
            return loss_fn(y, val_logits)

        #implement reduce learning rate on pleateu
        rlrop = {
            "factor":0.2,
            "patience":5,
            "min_lr":1e-9,
            "max_val":np.inf,
            "prec_step":None,
        }

        #implemet early_stopping
        es = {
            "min_delta":0.001,
            "patience":10,
            "max_val":np.inf,
            "prec_step":None,
        }
        values = {"loss":0.0, "val_loss":0.0}
        for epoch in range(epochs):
            start_time = time.time()
            #loss=0
            # Iterate over the batches of the dataset.
            step = 0
            values["loss"]=0.0
            for (x_batch_train, y_batch_train) in train_dataset:
                loss_value = train_step(x_batch_train, y_batch_train)
                values["loss"] +=loss_value
                # Log every 10 batches.
                if step % 10 == 0:
                    tf.get_logger().info("Training loss (for one batch) at step %d: %.4f",
                         (step, float(loss_value)))
                    tf.get_logger().info("Seen so far: %d samples" , ((step + 1) * batch_size))
                step+=1
            
            # Display metrics at the end of each epoch.
            train_acc = train_acc_metric.result()
            tf.get_logger().info("Training acc over epoch: %.4f" , (float(train_acc),))
            

            # Reset training metrics at the end of each epoch
            train_acc_metric.reset_states()
            values["val_loss"] = 0.0
            # Run a validation loop at the end of each epoch.
            if val:
                for x_batch_val, y_batch_val in val_dataset:
                    values["val_loss"] += test_step(x_batch_val, y_batch_val)

                val_acc = val_acc_metric.result()
                val_acc_metric.reset_states()

                tf.get_logger().info("Validation acc: %.4f"  ,(float(val_acc),))
            tf.get_logger().info("Time taken: %.2fs" , (time.time() - start_time))

            # At the end of the epoch I have to call my callbacks
            if rlrop["max_val"]<values[monitor_val]:
                rlrop["prec_step"]+=1
            else:
                rlrop["prec_step"]=0
                rlrop["max_val"]=values[monitor_val]

            if es["max_val"]<values[monitor_val]:
                es["prec_step"]+=1
            else:
                es["prec_step"]=0
                es["max_val"]=values[monitor_val]

            if rlrop["prec_step"]>rlrop["patience"]:
                newlr = max(rlrop["factor"]*self.model.optimizer.lr, rlrop["min_lr"])
                tf.keras.backend.set_value(self.model.optimizer.lr, newlr)

            if es["prec_step"]>es["patience"]:
                step=np.inf

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
        out_dir='./',
        complete=0
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
