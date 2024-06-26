#!/usr/bin/env python
__author__ = "Enzo Ubaldo Petrocco"
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

sys.path.insert(1, "../")
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
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


class StandardModels(GeneralModelClass):
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
        GeneralModelClass.__init__(self)
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

    def adv_custom_loss(self):
        """
        When model is embedded in adversarial training, we need to change adapt the loss function
        :return loss function (binary crossentropy)
        """
        def loss(y_true, y_pred):
            l = tf.keras.losses.binary_crossentropy(y_true[0], y_pred[0])
            return l

        return loss

    def DL_model_selection(
        self,
        TS,
        VS,
        adversary=0,
        eps=0.05,
        mult=0.2,
        learning_rates=[1e-5, 1e-4, 1e-3],
        batch_sizes=[2, 4, 8],
        gradcam=False,
        out_dir="./",
    ):
        """
        This function implements the model selection of Deep Learning model
        :param TS: Training set
        :param VS: Validation Set
        :param adversary: if enabled, adversarial training is enabled
        :param eps: if adversary enabled, step size of adversarial training
        :param mult: if adversary enabled, multiplier of adversarial training
        :param learning_rates: list of lr to be used for Model Selection
        :param batch_sizes: list of bs to be used for Model Selection
        :param gradcam: if enabled, gradcam callback is called
        :param out_dir: if gradcam enabled, output directory of gradcam heatmap
        """
        X = tf.stack(TS[0])
        y = tf.stack(TS[1])
        Xv = tf.stack(VS[0])
        yv = tf.stack(VS[1])
        best_loss = np.inf
        for lr in learning_rates:
            for bs in batch_sizes:
                with tf.device("/gpu:0"):
                    size = np.shape(TS[0][0])
                    input = Input(size, name="image")

                    resnet = ResNet50V2(
                        input_shape=size, weights="imagenet", include_top=False
                    )
                    fl = Flatten()(resnet.output)
                    out = (Dense(1, activation="sigmoid", name="dense"))(fl)

                    self.model = Model(inputs=resnet.input, outputs=out, name="model")

                    for layer in self.model.layers:
                        layer.trainable = False
                    for layer in self.model.layers[-2:]:
                        if not isinstance(layer, layers.BatchNormalization):
                            layer.trainable = True
                            if self.verbose_param:
                                print(f"Layer trainable name is: {layer.name}")
                    if adversary:
                        self.model = tf.keras.Sequential([input, self.model])
                    monitor_val = "val_loss"
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
                        patience=12,
                        verbose=self.verbose_param,
                        mode="auto",
                    )
                    adam = optimizers.Adam(lr)
                    optimizer = adam
                    self.model.compile(
                        loss="binary_crossentropy",
                        optimizer=optimizer,
                        metrics=["accuracy"],
                    )
                    # Wrap the model with adversarial regularization
                    if adversary:
                        adv_config = nsl.configs.make_adv_reg_config(
                            multiplier=mult, adv_step_size=eps
                        )
                        self.model = nsl.keras.AdversarialRegularization(
                            self.model, adv_config=adv_config, label_keys=["label"]
                        )
                        self.model.compile(
                            optimizer=optimizer,
                            metrics=["accuracy"],
                            loss=self.adv_custom_loss(),
                        )
                    else:
                        self.model.compile(
                            optimizer=optimizer,
                            metrics=["accuracy"],
                            loss="binary_crossentropy",
                        )

                    if adversary:
                        self.history = self.model.fit(
                            x={"image": X, "label": y},
                            epochs=self.epochs,
                            validation_data={"image": Xv, "label": yv},
                            callbacks=[early, lr_reduce],
                            verbose=self.verbose_param,
                            batch_size=bs,
                        )

                    else:
                        self.history = self.model.fit(
                            X,
                            y,
                            epochs=self.epochs,
                            validation_data=(Xv, yv),
                            callbacks=[early, lr_reduce],
                            verbose=self.verbose_param,
                            batch_size=bs,
                        )
                    if self.history.history[monitor_val][-1] < best_loss:
                        best_loss = self.history.history[monitor_val][-1]
                        best_bs = bs
                        best_lr = lr
                    self.model = None
                    del self.model
                    gc.collect()
        if self.verbose_param:
            print(f"Best bs={best_bs}; best lr={best_lr}, best loss={best_loss}")

        self.batch_size = best_bs
        self.learning_rate = best_lr
        self.model = None
        self.DL(TS, VS, adversary, eps, mult, gradcam, out_dir)

    def DL(self, TS, VS, adversary=0, eps=0.05, mult=0.2, gradcam=False, out_dir="./"):
        """
        This function implements the training of Deep Learning model
        :param TS: Training set
        :param VS: Validation Set
        :param adversary: if enabled, adversarial training is enabled
        :param eps: if adversary enabled, step size of adversarial training
        :param mult: if adversary enabled, multiplier of adversarial training
        :param learning_rates: lr to be used for training
        :param batch_sizes: bs to be used for training
        :param gradcam: if enabled, gradcam callback is called
        :param out_dir: if gradcam enabled, output directory of gradcam heatmap
        """
        with tf.device("/gpu:0"):
            X = tf.stack(TS[0])
            y = tf.stack(TS[1])
            Xv = tf.stack(VS[0])
            yv = tf.stack(VS[1])
            size = np.shape(TS[0][0])
            input = Input(size, name="image")

            resnet = ResNet50V2(input_shape=size, weights="imagenet", include_top=False)
            fl = Flatten()(resnet.output)
            out = (Dense(1, activation="sigmoid", name="dense"))(fl)

            self.model = Model(inputs=resnet.input, outputs=out, name="model")
            gradcam_layers = []
            for layer in self.model.layers:
                layer.trainable = False
            for layer in self.model.layers[-2:]:
                if not isinstance(layer, layers.BatchNormalization):
                    layer.trainable = True
                    gradcam_layers.append(layer.name)
                    if self.verbose_param:
                        print(f"Layer trainable name is: {layer.name}")
            if adversary:
                self.model = tf.keras.Sequential([input, self.model])
            lr_reduce = ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.1,
                patience=3,
                verbose=self.verbose_param,
                mode="max",
                min_lr=1e-9,
            )
            early = EarlyStopping(
                monitor="val_loss",
                min_delta=0.001,
                patience=12,
                verbose=self.verbose_param,
                mode="auto",
            )
            callbacks = [lr_reduce, early]
            adam = optimizers.Adam(self.learning_rate)
            optimizer = adam
            # Wrap the model with adversarial regularization.
            if adversary:
                adv_config = nsl.configs.make_adv_reg_config(
                    multiplier=mult, adv_step_size=eps
                )
                self.model = nsl.keras.AdversarialRegularization(
                    self.model, adv_config=adv_config
                )
                self.model.compile(
                    optimizer=optimizer,
                    metrics=["accuracy"],
                    loss=[self.adv_custom_loss()],
                )

            else:
                self.model.compile(
                    loss="binary_crossentropy",
                    optimizer=optimizer,
                    metrics=["accuracy"],
                )

            if adversary:
                self.history = self.model.fit(
                    x={"image": X, "label": y},
                    epochs=self.epochs,
                    validation_data={"image": Xv, "label": yv},
                    callbacks=callbacks,
                    verbose=self.verbose_param,
                    batch_size=self.batch_size,
                )
                self.model = Model(
                    inputs=self.model.layers[0].get_layer("model").layers[0].input,
                    outputs=self.model.layers[0].get_layer("model").layers[-1].output,
                )
            else:
                self.history = self.model.fit(
                    X,
                    y,
                    epochs=self.epochs,
                    validation_data=(Xv, yv),
                    callbacks=callbacks,
                    verbose=self.verbose_param,
                    batch_size=self.batch_size,
                )
            gc.collect()
            if gradcam:
                # Instantiation of the explainer
                for name in gradcam_layers:
                    for class_index in range(2):
                        # Call to explain() method
                        print("Before GradCAM explaining")
                        output = self.explain(
                            validation_data=(Xv, yv),
                            class_index=class_index,
                            layer_name=name,
                        )
                        # Save output
                        self.save(output, out_dir, name)

    def fit(
        self, TS, VS=None, adversary=0, eps=0.05, mult=0.2, gradcam=False, out_dir="./"
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
        """
        if self.type == "SVC":
            self.SVC(TS)
        elif self.type == "RFC":
            self.RFC(TS)
        elif self.type == "DL" or "RESNET":
            self.DL_model_selection(
                TS, VS, adversary, eps, mult, gradcam=gradcam, out_dir=out_dir
            )
        else:
            self.DL_model_selection(
                TS, VS, adversary, eps, mult, gradcam=gradcam, out_dir=out_dir
            )
