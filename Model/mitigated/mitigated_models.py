#!/usr/bin/env python
__author__ = "Enzo Ubaldo Petrocco"
import sys

sys.path.insert(1, "../../")
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Flatten, Input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import Model
from keras.applications.resnet_v2 import ResNet50V2
from keras import layers, optimizers
from Model.GeneralModel import GeneralModelClass
import neural_structured_learning as nsl
import gc


class MitigatedModels(GeneralModelClass):
    """
    
    """
    def __init__(
        self,
        type="DL",
        culture=0,
        verbose_param=0,
        epochs=15,
        batch_size=1,
        learning_rate=1e-3,
        lambda_index=-1,
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

    def custom_loss(self, out):
        """
        This function implements the loss and the regularizer of the mitigation stratyegy
        :param out: related to the corresponding output to be optimized
        :return loss function
        """
        def loss(y_true, y_pred):
            weights1 = self.model.layers[-3].kernel
            weights2 = self.model.layers[-2].kernel
            weights3 = self.model.layers[-1].kernel
            mean = tf.math.add(weights1, weights2)
            mean = tf.math.add(mean, weights3)
            mean = tf.multiply(mean, 1 / 3)
            mean = tf.multiply(mean, self.lamb)
            if out == 0:
                dist = tf.norm(weights1 - mean, ord="euclidean")
            if out == 1:
                dist = tf.norm(weights2 - mean, ord="euclidean")
            if out == 2:
                dist = tf.norm(weights3 - mean, ord="euclidean")
            dist = tf.multiply(dist, dist)
            loss = tf.keras.losses.binary_crossentropy(y_true[:,1], y_pred[:])
            res = tf.math.add(loss, dist)
            mask = tf.reduce_all(tf.equal(y_true[0][0], out))
            if not mask:
                return 0.0
            else:
                return res
        return loss

    def adv_custom_loss(self):
        """
        This function implements the loss and the regularizer of the mitigation stratyegy with adversarial enabled
        :return loss
        """
        def loss(y_true, y_pred):
            sum = 0.0
            for out in range(3):
                weights = self.model.layers[0].layers[0].layers[-1].kernel
                weights1 = weights[:, 0]
                weights2 = weights[:, 1]
                weights3 = weights[:, 2]
                mean = tf.math.add(weights1, weights2)
                mean = tf.math.add(mean, weights3)
                mean = tf.multiply(mean, 1 / 3)
                mean = tf.multiply(mean, self.lamb)
                if out == 0:
                    dist = tf.norm(weights1 - mean, ord="euclidean")
                if out == 1:
                    dist = tf.norm(weights2 - mean, ord="euclidean")
                if out == 2:
                    dist = tf.norm(weights3 - mean, ord="euclidean")
                dist = tf.multiply(dist, dist)
                loss = tf.keras.losses.binary_crossentropy(y_true[:,1], y_pred[:,out])
                res = tf.math.add(loss, dist)
                mask = tf.reduce_all(tf.equal(y_true[0][0], out))
                if mask:
                    sum += res
            return sum

        return loss

    def DL_model_selection(
        self,
        TS,
        VS,
        adversarial=0,
        eps=0.05,
        mult=0.2,
        learning_rates=[1e-5, 1e-4, 1e-3],
        batch_sizes=[1],
        out_dir="./",
        gradcam = False
    ):
        """
        This function implements the model selection of Deep Learning model with Mitigation Stategy
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

                    if adversarial:
                        y_0 = (Dense(3, activation="sigmoid", name="dense"))(fl)
                        self.model = Model(inputs=resnet.input, outputs=y_0, name="model")
                    else:
                        y1 = (Dense(1, activation="sigmoid", name="dense_0"))(fl)
                        y2 = (Dense(1, activation="sigmoid", name="dense_1"))(fl)
                        y3 = (Dense(1, activation="sigmoid", name="dense_2"))(fl)
                        self.model = Model(
                            inputs=resnet.input, outputs=[y1, y2, y3], name="model"
                        )

                    gradcam_layers = []
                    if adversarial:
                        for layer in self.model.layers:
                            layer.trainable = False
                        self.model.layers[-1].trainable = True
                        gradcam_layers.append(self.model.layers[-1].name)
                        # print(f"Layer trainable name is: {self.model.layers[-1].name}")
                    else:
                        for layer in self.model.layers:
                            layer.trainable = False
                        for layer in self.model.layers[-3:]:
                            if not isinstance(layer, layers.BatchNormalization):
                                if self.verbose_param:
                                    print(f"Layer trainable name is: {layer.name}")
                                layer.trainable = True
                                gradcam_layers.append(layer.name)

                    if adversarial:
                        self.model = tf.keras.Sequential([input, self.model])


                    monitor_val = f"val_loss"
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

                    # Wrap the model with adversarial regularization.
                    if adversarial:
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
                        # for layer in self.model.layers:
                        #    layer.summary()
                    else:
                        self.model.compile(
                            optimizer=optimizer,
                            metrics=["accuracy"],
                            loss=[
                                self.custom_loss(out=0),
                                self.custom_loss(out=1),
                                self.custom_loss(out=2),
                            ],
                        )
                        # self.model.summary()

                    tf.get_logger().setLevel("ERROR")



                    if adversarial:
                        self.history = self.model.fit(
                            x={"image": X, "label": y},
                            epochs=self.epochs,
                            validation_data={"image": Xv, "label": yv},
                            callbacks=[early, lr_reduce],
                            verbose=self.verbose_param,
                            batch_size=bs,
                        )

                    else:
                        # print(np.linalg.norm(np.array([i[0] for i in self.model.layers[len(self.model.layers)-2].get_weights()])-np.array([i[0] for i in self.model.layers[len(self.model.layers)-1].get_weights()])))
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
        self.DL(TS, VS, adversarial, eps, mult, gradcam=gradcam, out_dir=out_dir)

    def DL(
        self, TS, VS, adversarial=0, eps=0.05, mult=0.2, gradcam=False, out_dir="./"
    ):
        """
        This function implements the training of Deep Learning model with Mitigation Strategy
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
        X = tf.stack(TS[0])
        y = tf.stack(TS[1])
        Xv = tf.stack(VS[0])
        yv = tf.stack(VS[1])
        with tf.device("/gpu:0"):
            self.model = None

            size = np.shape(TS[0][0])
            input = Input(size, name="image")

            resnet = ResNet50V2(
                input_shape=size, weights="imagenet", include_top=False
            )
            fl = Flatten()(resnet.output)

            if adversarial:
                y_0 = (Dense(3, activation="sigmoid", name="dense"))(fl)
                self.model = Model(inputs=resnet.input, outputs=y_0, name="model")
            else:
                y1 = (Dense(1, activation="sigmoid", name="dense_0"))(fl)
                y2 = (Dense(1, activation="sigmoid", name="dense_1"))(fl)
                y3 = (Dense(1, activation="sigmoid", name="dense_2"))(fl)
                self.model = Model(
                    inputs=resnet.input, outputs=[y1, y2, y3], name="model"
                )

            gradcam_layers = []
            if adversarial:
                for layer in self.model.layers:
                    layer.trainable = False
                self.model.layers[-1].trainable = True
                gradcam_layers.append(self.model.layers[-1].name)
                # print(f"Layer trainable name is: {self.model.layers[-1].name}")
            else:
                for layer in self.model.layers:
                    layer.trainable = False
                for layer in self.model.layers[-3:]:
                    if not isinstance(layer, layers.BatchNormalization):
                        if self.verbose_param:
                            print(f"Layer trainable name is: {layer.name}")
                        layer.trainable = True
                        gradcam_layers.append(layer.name)

            if adversarial:
                self.model = tf.keras.Sequential([input, self.model])



            monitor_val = f"val_loss"
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
            adam = optimizers.Adam(self.learning_rate)
            optimizer = adam

            #tf.get_logger().setLevel("ERROR")
            # Wrap the model with adversarial regularization.
            if adversarial:
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
                self.history = self.model.fit(
                    x={"image": X, "label": y},
                    epochs=self.epochs,
                    validation_data={"image": Xv, "label": yv},
                    callbacks=[early, lr_reduce],
                    verbose=self.verbose_param,
                    batch_size=self.batch_size,
                )

                self.model = Model(
                    inputs=self.model.layers[0].get_layer("model").layers[0].input,
                    outputs=self.model.layers[0].get_layer("model").layers[-1].output,
                )
            else:
                self.model.compile(
                    optimizer=optimizer,
                    metrics=["accuracy"],
                    loss=[
                        self.custom_loss(out=0),
                        self.custom_loss(out=1),
                        self.custom_loss(out=2),
                    ],
                )
                # print(np.linalg.norm(np.array([i[0] for i in self.model.layers[len(self.model.layers)-2].get_weights()])-np.array([i[0] for i in self.model.layers[len(self.model.layers)-1].get_weights()])))
                self.history = self.model.fit(
                    X,
                    y,
                    epochs=self.epochs,
                    validation_data=(Xv, yv),
                    callbacks=[early, lr_reduce],
                    verbose=self.verbose_param,
                    batch_size=self.batch_size,
                )

            

                
            if gradcam:
                # Instantiation of the explainer
                for name in gradcam_layers:
                    for class_index in range(2):
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
        if self.type == "DL" or "RESNET":
            self.DL_model_selection(
                TS, VS, adversary, eps, mult, gradcam=gradcam, out_dir=out_dir
            )
        else:
            self.DL_model_selection(
                TS, VS, adversary, eps, mult, gradcam=gradcam, out_dir=out_dir
            )
