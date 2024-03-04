import sys

sys.path.insert(1, "../../")
import numpy as np
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
from matplotlib import pyplot as plt
import random
import gc
from tf_explain.callbacks.grad_cam import GradCAMCallback

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
    ):
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
        def loss(y_true, y_pred):
            weights1 = self.model.layers[len(self.model.layers) - 1].kernel
            weights2 = self.model.layers[len(self.model.layers) - 2].kernel
            weights3 = self.model.layers[len(self.model.layers) - 3].kernel
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
            loss = tf.keras.losses.binary_crossentropy(y_true[0][1], y_pred[0])
            res = tf.math.add(loss, dist)
            mask = tf.reduce_all(tf.equal(y_true[0][0], out))
            if not mask:
                return 0.0
            else:
                return res

        return loss

    def adv_custom_loss(self):
        def loss(y_true, y_pred):
            sum = 0.0
            for out in range(3):
                weights = self.model.layers[0].layers[len(self.model.layers) - 2].kernel
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
                loss = tf.keras.losses.binary_crossentropy(y_true[0][1], y_pred[0])
                res = tf.math.add(loss, dist)
                mask = tf.reduce_all(tf.equal(y_true[0][0], out))
                if mask:
                    sum += res
            return sum

        return loss
    
    def DL_model_selection(self, TS, VS, adversarial=0, eps=0.05, mult=0.2, learning_rates=[1e-5, 1e-4, 1e-3], batch_sizes=[2,4,8], out_dir="./"):
        best_loss = np.inf
        for lr in learning_rates:
            for bs in batch_sizes:
                with tf.device("/gpu:0"):
                    size = np.shape(TS[0][0])
                    input = Input(size, name="image")
                    x = tf.keras.Sequential(
                        [ResNet50V2(input_shape=size, weights="imagenet", include_top=False)]
                    )(input)
                                 
                    x = Flatten()(x)
                    if adversarial:
                        y = (Dense(3, activation="sigmoid", name="dense"))(x)
                        self.model = Model(inputs=input, outputs=y, name="model")
                    else:
                        y1 = (Dense(1, activation="sigmoid", name="dense_0"))(x)
                        y2 = (Dense(1, activation="sigmoid", name="dense_1"))(x)
                        y3 = (Dense(1, activation="sigmoid", name="dense_2"))(x)
                        self.model = Model(inputs=input, outputs=[y1, y2, y3], name="model")
                    self.model.trainable = True
                    for layer in self.model.layers[1].layers:
                        layer.trainable = False
                    # Last layer only one trainable
                    if adversarial:
                        self.model.layers[-1].trainable = True
                        #print(f"Layer trainable name is: {self.model.layers[-1].name}")
                    else:
                        for layer in self.model.layers[-3:]:
                            if not isinstance(layer, layers.BatchNormalization):
                                if self.verbose_param:
                                    print(f"Layer trainable name is: {layer.name}")
                                layer.trainable = True
                    
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
                        #for layer in self.model.layers:
                        #    layer.summary()
                    else:
                        self.model.compile(
                            optimizer=optimizer,
                            metrics=["accuracy"],
                            loss=[
                                self.custom_loss(out=0),
                                self.custom_loss(out=1),
                                self.custom_loss(out=2),
                            ],)
                        #self.model.summary()

                    tf.get_logger().setLevel("ERROR")

                    X = tf.stack(TS[0])
                    y = tf.stack(TS[1])
                    Xv = tf.stack(VS[0])
                    yv = tf.stack(VS[1])
                    
                    if adversarial:
                        self.history = self.model.fit(
                            x={"image": X, "label": y},
                            epochs=self.epochs,
                            validation_data={"image": Xv, "label": yv},
                            callbacks=[early, lr_reduce],
                            verbose=self.verbose_param,
                            batch_size=bs,
                        )
                        self.model = self.model.layers[0]
                    else:
                        #print(np.linalg.norm(np.array([i[0] for i in self.model.layers[len(self.model.layers)-2].get_weights()])-np.array([i[0] for i in self.model.layers[len(self.model.layers)-1].get_weights()])))
                        self.history = self.model.fit(
                            X,
                            y,
                            epochs=self.epochs,
                            validation_data=(Xv, yv),
                            callbacks=[early, lr_reduce],
                            verbose=self.verbose_param,
                            batch_size=bs,
                        )
                    if self.history.history[monitor_val][-1]<best_loss:
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
        self.DL(TS, VS, adversarial, eps, mult, out_dir)  
            

    def DL(self, TS, VS, adversarial=0, eps=0.05, mult=0.2, gradcam = False,  out_dir="./"):
        with tf.device("/gpu:0"):
            X = tf.stack(TS[0])
            y = tf.stack(TS[1])
            Xv = tf.stack(VS[0])
            yv = tf.stack(VS[1])

            size = np.shape(TS[0][0])
            input = Input(size, name="image")
            x = tf.keras.Sequential(
                [ResNet50V2(input_shape=size, weights="imagenet", include_top=False)]
            )(input)
            
            
            x = Flatten()(x)
            if adversarial:
                y = (Dense(3, activation="sigmoid", name="dense"))(x)
                self.model = Model(inputs=input, outputs=y, name="model")
            else:
                y1 = (Dense(1, activation="sigmoid", name="dense_0"))(x)
                y2 = (Dense(1, activation="sigmoid", name="dense_1"))(x)
                y3 = (Dense(1, activation="sigmoid", name="dense_2"))(x)
                self.model = Model(inputs=input, outputs=[y1, y2, y3], name="model")
            self.model.trainable = True
            for layer in self.model.layers[1].layers:
                layer.trainable = False
            # Last layer only one trainable
            gradcam_layers = []
            if adversarial:
                self.model.layers[-1].trainable = True
                gradcam_layers.append(self.model.layers[-1].name)
                #print(f"Layer trainable name is: {self.model.layers[-1].name}")
            else:
                for layer in self.model.layers[-3:]:
                    if not isinstance(layer, layers.BatchNormalization):
                        if self.verbose_param:
                                    print(f"Layer trainable name is: {layer.name}")
                        layer.trainable = True
                        gradcam_layers.append(layer.name)

            
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
            grad1 = GradCAMCallback(
                validation_data=(Xv, yv),
                class_index=1,
                output_dir=out_dir + 'class0',
                )
            adam = optimizers.Adam(self.learning_rate)
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
                #for layer in self.model.layers:
                #    layer.summary()
            else:
                self.model.compile(
                    optimizer=optimizer,
                    metrics=["accuracy"],
                    loss=[
                        self.custom_loss(out=0),
                        self.custom_loss(out=1),
                        self.custom_loss(out=2),
                    ],)
                
            tf.get_logger().setLevel("ERROR")
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=out_dir, histogram_freq=1)
            if gradcam:
                callbacks = [early, lr_reduce, tensorboard_callback]
                for name in gradcam_layers:
                    grad_call = GradCAMCallback(
                        validation_data=(Xv, yv),
                        class_index=0,
                        output_dir=out_dir + 'gradcam_' + name,
                        layer_name=name
                        )
                    callbacks.append(grad_call)
            else:
                callbacks = [early, lr_reduce, tensorboard_callback]
            if adversarial:
                self.history = self.model.fit(
                    x={"image": X, "label": y},
                    epochs=self.epochs,
                    validation_data={"image": Xv, "label": yv},
                    callbacks=callbacks,
                    verbose=self.verbose_param,
                    batch_size=self.batch_size,
                )
                self.model = self.model.layers[0]
            else:
                #print(np.linalg.norm(np.array([i[0] for i in self.model.layers[len(self.model.layers)-2].get_weights()])-np.array([i[0] for i in self.model.layers[len(self.model.layers)-1].get_weights()])))
                self.history = self.model.fit(
                    X,
                    y,
                    epochs=self.epochs,
                    validation_data=(Xv, yv),
                    callbacks=callbacks,
                    verbose=self.verbose_param,
                    batch_size=self.batch_size,
                )

                #print(np.linalg.norm(np.array([i[0] for i in self.model.layers[len(self.model.layers)-2].get_weights()])-np.array([i[0] for i in self.model.layers[len(self.model.layers)-1].get_weights()])))
        

    def fit(self, TS, VS=None, adversary=0, eps=0.05, mult=0.2, gradcam=False, out_dir="./"):
        if self.type == "DL" or "RESNET":
            self.DL_model_selection(TS, VS, adversary, eps, mult, gradcam=gradcam, out_dir=out_dir)
        else:
            self.DL_model_selection(TS, VS, adversary, eps, mult, gradcam=gradcam, out_dir=out_dir)
