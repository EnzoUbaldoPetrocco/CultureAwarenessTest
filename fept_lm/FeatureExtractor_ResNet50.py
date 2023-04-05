#! /usr/bin/env python3


import cv2
import keras.layers as L
import numpy as np
import tensorflow as tf
from keras import Model
from keras import backend as K
from keras import layers, models, optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense, Dropout, Flatten, Input, Lambda, MaxPooling3D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from skimage.color import gray2rgb
from tensorflow.keras.applications import efficientnet
from tensorflow.keras.applications.efficientnet import EfficientNetB3
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2S
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

import manipulating_images_better

'''config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)'''
import os


BATCH_SIZE = 1


def to_grayscale_then_rgb(image):
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.grayscale_to_rgb(image)
    return image


class FeatureExtractor:
    def __init__(self, ds_selection = ""):

        self.ds_selection = ds_selection
        itd = manipulating_images_better.ImagesToData(ds_selection = self.ds_selection)
        itd.bf_ml()
        itd.little_mix()

        MCX = itd.MCX
        CXT = itd.CXT
        MCY = itd.MCY
        CYT = itd.CYT

        MFX = itd.MFX
        FXT = itd.FXT
        MFY = itd.MFY
        FYT = itd.FYT

        MXT = itd.MXT
        MYT = itd.MYT

        batch_size = 1

        validation_split = 0.1
        
        chindatagen = ImageDataGenerator(
            validation_split=validation_split,
            #rescale=1/255,
    preprocessing_function=to_grayscale_then_rgb)

        chinvaldatagen = ImageDataGenerator(
            validation_split=validation_split,
            #rescale=1/255,
    preprocessing_function=to_grayscale_then_rgb)

        frendatagen = ImageDataGenerator(
            validation_split=validation_split,
            #rescale=1/255,
    preprocessing_function=to_grayscale_then_rgb)

        frenvaldatagen = ImageDataGenerator(
            validation_split=validation_split,
            #rescale=1/255,
    preprocessing_function=to_grayscale_then_rgb)

        ######################################################################################
        ############################# MODEL GENERATION #######################################
        
        model = tf.keras.Sequential([
        ResNet50( #tf.keras.applications.efficientnet.EfficientNetB0
            input_shape=(itd.size, itd.size, 3),
            weights='imagenet',
            include_top=False
        )#,
        #L.GlobalAveragePooling2D()#,
        #L.Dense(1, activation='sigmoid')
    ])
        
        model.add(Flatten())
        model.add(Dropout(0.15))
        model.add(Dense(50, activation = 'relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.trainable = True
        for layer in model.layers[0].layers:
            layer.trainable = False
        # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
        for layer in model.layers[0].layers[-1:]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True
        #model.summary()

        ####################################################################################
        ###################### TRAINING LAST LAYERS AND FINE TUNING ########################
        print('RETRAINING')
        
        ep = 100
        eps_fine = 10
        verbose_param = 1
        
        lr_reduce = ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=3, verbose=1, mode='max', min_lr=1e-8)
        #checkpoint = ModelCheckpoint('vgg16_finetune.h15', monitor= 'val_accuracy', mode= 'max', save_best_only = True, verbose= 0)
        early = EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=10, verbose=1, mode='auto')
        
        learning_rate= 4e-4
        learning_rate_fine = 1e-8
        
        adam = optimizers.Adam(learning_rate)
        sgd = tf.keras.optimizers.SGD(learning_rate)
        rmsprop = tf.keras.optimizers.RMSprop(learning_rate)
        adadelta = tf.keras.optimizers.Adadelta(learning_rate)
        adagrad = tf.keras.optimizers.Adagrad(learning_rate)
        adamax = tf.keras.optimizers.Adamax(learning_rate)

        optimizer = adam
        fine_optimizer = optimizers.SGD(learning_rate_fine)

        model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
        #history = model.fit(X_train, y_train, batch_size = 1, epochs=50, validation_data=(X_test,y_test), callbacks=[lr_reduce,checkpoint])
        
        if self.ds_selection == "chinese":
            print('chinese')
            chinese = chindatagen.flow_from_directory('../../FE/' + ds_selection + '/chinese',
            target_size = (itd.size, itd.size),
            batch_size = batch_size,
            color_mode = 'rgb',
            class_mode = 'binary',
            subset = 'training')

            chinese_val = chinvaldatagen.flow_from_directory('../../FE/' + ds_selection + '/chinese',
            target_size = (itd.size, itd.size),
            batch_size = batch_size,
            class_mode = 'binary',
            color_mode = 'rgb',
            subset = 'validation')

            '''for i in chinese:
                plt.figure()
                plt.imshow(i[0][0])
                plt.show()'''

            

            Number_Of_Training_Images = chinese.classes.shape[0]
            steps_per_epoch = Number_Of_Training_Images/batch_size


            history = model.fit(chinese, 
            #batch_size = batch_size, 
            epochs=ep, validation_data=chinese_val, 
            #steps_per_epoch = steps_per_epoch,
            callbacks=[early, lr_reduce],verbose=verbose_param)
            
        if self.ds_selection == "french":
            print('french')
            french = frendatagen.flow_from_directory('../../FE/' + ds_selection + '/french',
            target_size = (itd.size, itd.size),
            batch_size = batch_size,
            class_mode = 'binary',
            color_mode = 'rgb',
            subset = 'training')

            french_val = frenvaldatagen.flow_from_directory('../../FE/' + ds_selection + '/french',
            target_size = (itd.size, itd.size),
            batch_size = batch_size,
            class_mode = 'binary',
            color_mode = "rgb",
            subset = 'validation')

            Number_Of_Training_Images = french.classes.shape[0]
            steps_per_epoch = Number_Of_Training_Images/batch_size

            history = model.fit(french, epochs=ep, 
            validation_data=french_val, callbacks=[early, lr_reduce], verbose=verbose_param)
            
            
        if self.ds_selection == "mix":
            print('mix')
            dataset = chindatagen.flow_from_directory('../../FE/' + ds_selection,
            target_size = (itd.size, itd.size),
            batch_size = batch_size,
            class_mode = 'binary',
            color_mode = "rgb",
            subset = 'training')

            dataset_val = chinvaldatagen.flow_from_directory('../../FE/' + ds_selection,
            target_size = (itd.size, itd.size),
            batch_size = batch_size,
            class_mode = 'binary',
            color_mode = "rgb",
            subset = 'validation')

            
            Number_Of_Training_Images = dataset.classes.shape[0]
            steps_per_epoch = Number_Of_Training_Images/batch_size
            history = model.fit(dataset,  
            steps_per_epoch = steps_per_epoch,
            epochs=ep, validation_data=dataset_val, callbacks=[early, lr_reduce], verbose=verbose_param)

            
        ##############################################################
        ############## PLOT SOME RESULTS ############################
        plot = False
        if plot:
            train_acc = history.history['accuracy']
            val_acc = history.history['val_accuracy']
            train_loss = history.history['loss']
            val_loss = history.history['val_loss']
            No_Of_Epochs = range(ep)
            train_acc_x = range(len(train_acc))
            val_acc_x = range(len(train_acc))
            train_loss_x = range(len(train_acc))
            val_loss_x = range(len(train_acc))

            plt.plot(train_acc_x, train_acc, marker = 'o', color = 'blue', markersize = 10, 
                            linewidth = 1.5, label = 'Training Accuracy')
            plt.plot(val_acc_x, val_acc, marker = '.', color = 'red', markersize = 10, 
                            linewidth = 1.5, label = 'Validation Accuracy')

            plt.title('Training Accuracy and Testing Accuracy w.r.t Number of Epochs')

            plt.legend()

            plt.figure()

            plt.plot(train_loss_x, train_loss, marker = 'o', color = 'blue', markersize = 10, 
                            linewidth = 1.5, label = 'Training Loss')
            plt.plot(val_loss_x, val_acc, marker = '.', color = 'red', markersize = 10, 
                            linewidth = 1.5, label = 'Validation Loss')

            plt.title('Training Loss and Testing Loss w.r.t Number of Epochs')

            plt.legend()

            plt.show()

                

        #################################################
        ############# FEATURE EXTRACTION ################
        #print(model.layers[-2])
        #model = Model(inputs=model.inputs, outputs=model.layers[:-2])
        model.layers.pop()        
        
        print('FEATURE EXTRACTION')
        features = []
        for i in MCX:
            x = np.reshape(i, (itd.size,itd.size))*255
            x = cv2.merge([x,x,x])
            x = image.img_to_array(x)
            x = np.expand_dims(x, axis=0)
            feature = model.predict(x, verbose = 0)
            features.append(feature[0].flatten())

            
        self.MCX = np.array(features)
        self.MCY = MCY

        features = []
        for i in CXT:
            x = np.reshape(i, (itd.size,itd.size))*255
            x = cv2.merge([x,x,x])
            x = image.img_to_array(x)
            x = np.expand_dims(x, axis=0)
            feature = model.predict(x, verbose = 0)
            features.append(feature[0].flatten())
        
        self.CXT = np.array(features)
        self.CYT = CYT

        features = []
        for i in MFX:
            x = np.reshape(i, (itd.size,itd.size))*255
            x = cv2.merge([x,x,x])
            x = image.img_to_array(x)
            x = np.expand_dims(x, axis=0)
            feature = model.predict(x, verbose = 0)
            features.append(feature[0].flatten())
        
        self.MFX = np.array(features)
        self.MFY = MFY

        features = []
        for i in FXT:
            x = np.reshape(i, (itd.size,itd.size))*255
            x = cv2.merge([x,x,x])
            x = image.img_to_array(x)
            x = np.expand_dims(x, axis=0)
            feature = model.predict(x, verbose = 0)
            features.append(feature[0].flatten())
        
        self.FXT = np.array(features)
        self.FYT = FYT

        features = []
        for i in MXT:
            x = np.reshape(i, (itd.size,itd.size))*255
            x = cv2.merge([x,x,x])
            x = image.img_to_array(x)
            x = np.expand_dims(x, axis=0)
            feature = model.predict(x, verbose = 0)
            features.append(feature[0].flatten())
        
        self.MXT = np.array(features)
        self.MYT = MYT

        

        