#!/usr/bin/env python
__author__ = "Enzo Ubaldo Petrocco"

import os
import gc
import sys
import random
import copy
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from datetime import datetime
from math import ceil
from Model.GeneralModel import GeneralModelClass

# Reproducibility
random.seed(datetime.now().timestamp())
tf.random.set_seed(datetime.now().timestamp())

class NullWriter:
    def write(self, _): pass

def suppress_output():
    sys.stdout = NullWriter()

def restore_output():
    sys.stdout = sys.__stdout__

class AdversarialStandard(GeneralModelClass):
    def __init__(self, type="RESNET", points=50, kernel="linear", verbose_param=0,
                 learning_rate=1e-3, epochs=15, batch_size=1, weights=None,
                 imbalanced=0, class_division=0, only_imb_imgs=0,
                 save_discriminator=0, path='./', culture=0):
        
        GeneralModelClass.__init__(self, standard=1, adversarial=1, imbalanced=imbalanced)
        
        self.type = type
        self.points = points
        self.kernel = kernel
        self.verbose_param = verbose_param
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.class_division = class_division
        self.only_imb_imgs = only_imb_imgs
        self.save_discriminator = save_discriminator
        self.path = path
        self.culture = culture
        self.weights = weights if weights is not None else np.ones(self.n_cultures)
        self.model = None

    @tf.function
    def generate_adversarial_image_pgd(self, img, lbl, model, epsilon=0.1, alpha=0.002, num_iter=50):
        img_batch = tf.expand_dims(img, axis=0)
        lbl_batch = tf.expand_dims(lbl, axis=0)
        x_adv = tf.identity(img_batch) 

        for _ in range(num_iter):
            with tf.GradientTape() as tape:
                tape.watch(x_adv)
                prediction = model(x_adv, training=False)
                loss = tf.keras.losses.categorical_crossentropy(lbl_batch, prediction)
            
            gradients = tape.gradient(loss, x_adv)
            # Logic: Perturb in 0-255 space, but considering epsilon is usually defined for 0-1
            perturbations = alpha * 255.0 * tf.sign(gradients)
            x_adv = x_adv + perturbations
            
            # Project and Clip
            x_adv = tf.clip_by_value(x_adv, img_batch - (epsilon * 255.0), img_batch + (epsilon * 255.0))
            x_adv = tf.clip_by_value(x_adv, 0, 255.0)
            
        return x_adv

    def plot_images(self, generated_images, j=0, num_rows=3, num_cols=6):
        if not generated_images: return
        generated_images = generated_images[0:num_rows * num_cols]
        fig = plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0))
        for i, img in enumerate(generated_images):
            plt.subplot(num_rows, num_cols, i + 1)
            plt.imshow(img / 255.0)
            plt.axis("off")
        
        plt.tight_layout()
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        plt.savefig(os.path.join(self.path, f"class={j}.jpg"))
        plt.close()

    def remove_data_aug(self, model):
        # Assumes standard stack: Input -> Aug -> Scaling -> Base -> ...
        # Skips layer 1 (Augmentation) and starts from Scaling/Base
        inputs = keras.Input(shape=self.shape)
        x = inputs
        # Dynamically rebuild layers excluding the Augmentation sequence
        for layer in model.layers[2:]:
            x = layer(x)
        return keras.Model(inputs=inputs, outputs=x)

    def LearningAdversarially(self, TS, VS, aug, path="./", eps=0.1, **kwargs):
        # 1. Shuffle
        idx_t = np.random.permutation(len(TS[0]))
        TS = ([TS[0][i] for i in idx_t], [TS[1][i] for i in idx_t])
        if VS:
            idx_v = np.random.permutation(len(VS[0]))
            VS = ([VS[0][i] for i in idx_v], [VS[1][i] for i in idx_v])

        if self.imbalanced:
            TS = self.ImbalancedTransformation(TS)

        TS0 = copy.deepcopy(TS)
        adversarial_models = []

        # 2. Phase 1: Train Discriminators (Either one global or per-class)
        if self.class_division:
            for j in range(2): 
                tempTS = ([TS[0][i] for i in range(len(TS[0])) if TS[1][i][self.n_cultures] == j],
                          [TS[1][i] for i in range(len(TS[1])) if TS[1][i][self.n_cultures] == j])
                tempVS = ([VS[0][i] for i in range(len(VS[0])) if VS[1][i][self.n_cultures] == j],
                          [VS[1][i] for i in range(len(VS[1])) if VS[1][i][self.n_cultures] == j])
                
                self.ModelSelection(TS=tempTS, VS=tempVS, aug=aug, adv=1, eps=eps, path=path, **kwargs)
                adversarial_models.append(self.remove_data_aug(self.model) if aug else self.model)
                self.model = None
                gc.collect()
        else:
            self.ModelSelection(TS=TS, VS=VS, aug=aug, adv=1, eps=eps, path=path, **kwargs)
            adversarial_models = self.remove_data_aug(self.model) if aug else self.model

        # 3. Generate Adversarial Samples for Training Set
        images_to_plot = []
        imgs, ys = TS[0], TS[1]
        for i in range(len(imgs) // 8):
            target_model = adversarial_models[int(ys[i][self.n_cultures])] if self.class_division else adversarial_models
            lbl = tf.cast(ys[i][0:self.n_cultures], dtype=tf.float32)
            adv_img = self.generate_adversarial_image_pgd(tf.cast(imgs[i], tf.float32), lbl, target_model, epsilon=eps)[0]
            
            TS[0].append(adv_img.numpy())
            TS[1].append(ys[i])
            if i < 18: images_to_plot.append(adv_img.numpy())

        self.plot_images(images_to_plot, j=-1 if not self.class_division else 0)

        # 4. Phase 2: Final Training on Augmented Data
        self.ModelSelection(TS=TS, VS=VS, aug=aug, adv=0, eps=eps, path=path, **kwargs)
        tf.keras.backend.clear_session()

    def ModelSelection(self, TS, VS, aug, batches=[32], lrs=[1e-3], fine_lrs=[1e-5], epochs=15, fine_epochs=5, nDropouts=[0.3], adv=0, **kwargs):
        best_loss = np.inf
        best_params = {}

        for b in batches:
            for lr in lrs:
                for f_lr in fine_lrs:
                    for drop in nDropouts:
                        loss = self.DL(TS, VS, aug=aug, batch_size=b, lr=lr, fine_lr=f_lr, 
                                       epochs=epochs, fine_epochs=fine_epochs, nDropout=drop, adv=adv, **kwargs)
                        if loss < best_loss:
                            best_loss = loss
                            best_params = {'b': b, 'lr': lr, 'f_lr': f_lr, 'drop': drop}
        
        # Retrain with best params on combined TS+VS (as per original logic)
        self.DL(TS, VS, aug=aug, batch_size=best_params['b'], lr=best_params['lr'], 
                fine_lr=best_params['f_lr'], epochs=epochs, fine_epochs=fine_epochs, 
                nDropout=best_params['drop'], val=False, adv=adv, **kwargs)

    def ImbalancedTransformation(self, TS):
        newX, newY = [], []
        for img, label in zip(TS[0], TS[1]):
            # Get index of active class
            class_idx = np.argmax(label[:self.n_cultures])
            repeat = ceil(1.0 / self.weights[class_idx])
            for _ in range(repeat):
                newX.append(img)
                newY.append(label)
        return (newX, newY)

    def DL(self, TS, VS, aug=False, batch_size=32, lr=1e-3, fine_lr=1e-5, epochs=1, fine_epochs=1, nDropout=0.2, g=0.1, val=True, adv=0, **kwargs):
        shape = np.shape(TS[0][0])
        self.shape = shape
        monitor = "val_loss" if val else "loss"

        # Data Pipeline
        def map_fn(img, y):
            label = y[0:self.n_cultures] if adv else y[self.n_cultures]
            return img, label

        train_ds = tf.data.Dataset.from_tensor_slices(TS).map(map_fn).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_ds = tf.data.Dataset.from_tensor_slices(VS).map(map_fn).batch(batch_size).prefetch(tf.data.AUTOTUNE) if val else None

        # Build Model
        base_model = keras.applications.ResNet50V2(weights="imagenet", include_top=False, input_shape=shape)
        base_model.trainable = False

        inputs = keras.Input(shape=shape)
        x = inputs
        if aug:
            x = layers.RandomFlip("horizontal")(x)
            x = layers.RandomRotation(0.01)(x)
            x = layers.GaussianNoise(g)(x)
        x = layers.Rescaling(1./255.0)(x)
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(nDropout)(x)
        
        output_dim = self.n_cultures if adv else 1
        activation = "softmax" if adv else "sigmoid"
        outputs = layers.Dense(output_dim, activation=activation)(x)
        
        self.model = keras.Model(inputs, outputs)

        # Compile and Fit
        loss_fn = keras.losses.CategoricalCrossentropy() if adv else keras.losses.BinaryCrossentropy()
        self.model.compile(optimizer=keras.optimizers.Adam(lr), loss=loss_fn, metrics=['accuracy'])
        
        callbacks = [EarlyStopping(monitor=monitor, patience=5), ReduceLROnPlateau(monitor=monitor, factor=0.2)]
        
        self.model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=callbacks, verbose=self.verbose_param)

        # Fine Tuning
        base_model.trainable = True
        self.model.compile(optimizer=keras.optimizers.Adam(fine_lr), loss=loss_fn, metrics=['accuracy'])
        history = self.model.fit(train_ds, epochs=fine_epochs, validation_data=val_ds, callbacks=callbacks, verbose=self.verbose_param)

        return history.history[monitor][-1]

    def fit(self, TS, VS=None, aug=0, g=0.1, eps=0.3, out_dir="./", **kwargs):
        if self.type in ["DL", "RESNET"]:
            self.LearningAdversarially(TS, VS, aug=aug, g=g, path=out_dir, eps=eps)