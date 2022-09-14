#! /usr/bin/env python3
from audioop import rms
from unicodedata import name
import manipulating_images_better
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import cv2
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import layers, models, Model, optimizers
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
import tensorflow as tf
from skimage.color import gray2rgb
from matplotlib import pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from keras.layers import Input, Lambda, Dense, Flatten,Dropout
from keras.models import Sequential
from keras import backend as K
from tensorflow.keras.applications import EfficientNetB0


BATCH_SIZE = 1

data_augmentation = keras.Sequential(
    [layers.RandomFlip("horizontal"), layers.RandomRotation(0.1),]
)

def to_grayscale_then_rgb(image):
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.grayscale_to_rgb(image)
    return image

def batch_generator(X, Y, batch_size = BATCH_SIZE):
    indices = np.arange(len(X)) 
    batch=[]
    while True:
            # it might be a good idea to shuffle your data before each epoch
            np.random.shuffle(indices) 
            for i in indices:
                batch.append(i)
                if len(batch)==batch_size:
                    yield X[batch], Y[batch]
                    batch=[]


class FeatureExtractor:
    def __init__(self, ds_selection = ""):

        self.ds_selection = ds_selection
        itd = manipulating_images_better.ImagesToData(ds_selection = self.ds_selection)
        itd.bf_ml()

        CX = itd.CX
        CXT = itd.CXT
        CY = itd.CY
        CYT = itd.CYT

        FX = itd.FX
        FXT = itd.FXT
        FY = itd.FY
        FYT = itd.FYT

        MX = itd.MX
        MXT = itd.MXT
        MY = itd.MY
        MYT = itd.MYT

        batch_size = 16
        batch_fit = 8

        validation_split = 0.1
                
        chindatagen = ImageDataGenerator(
            validation_split=validation_split,
            rescale=1/255,
    preprocessing_function=to_grayscale_then_rgb)

        chinvaldatagen = ImageDataGenerator(
            validation_split=validation_split,
            rescale=1/255,
    preprocessing_function=to_grayscale_then_rgb)

        frendatagen = ImageDataGenerator(
            validation_split=validation_split,
            rescale=1/255,
    preprocessing_function=to_grayscale_then_rgb)

        frenvaldatagen = ImageDataGenerator(
            validation_split=validation_split,
            rescale=1/255,
    preprocessing_function=to_grayscale_then_rgb)

        ######################################################################################
        ############################# MODEL GENERATION #######################################
        #model = VGG16(weights='imagenet', include_top=False,  input_shape=(itd.size,itd.size,3))
        base_model = tf.keras.applications.VGG19(input_shape=(itd.size,itd.size,3), # define the input shape
                                               include_top=False, # remove the classification layer
                                               pooling='avg',
                                               weights='imagenet') # use ImageNet pre-trained weights
        base_model.trainable = True

        for layer in base_model.layers[:len(base_model.layers)-4]:
            layer.trainable = False
        #base_model.summary()
        # Create the model
        model = Sequential()
        # Add the vgg convolutional base model
        model.add(base_model)
        model.add(Dense(800, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(8, activation='relu', name='feature_extractor'))
        #model.trainable = False
        model.add(Dense(1, activation='sigmoid'))
        # Show a summary of the model. Check the number of trainable parameters
        # Freeze four convolution blocks
        model.trainable = True

        ####################################################################################
        ###################### TRAINING LAST LAYERS AND FINE TUNING ########################
        print('RETRAINING')
        
        ep = 20
        eps_fine = 50
        verbose_param = 1
        
        lr_reduce = ReduceLROnPlateau(monitor='val_binary_accuracy', factor=0.2, patience=4, verbose=1, mode='max', min_lr=1e-8)
        #checkpoint = ModelCheckpoint('vgg16_finetune.h15', monitor= 'val_accuracy', mode= 'max', save_best_only = True, verbose= 0)
        early = EarlyStopping(monitor='val_binary_accuracy', min_delta=0.001, patience=20, verbose=1, mode='auto')
        
        learning_rate= 2.5e-4
        learning_rate_fine = 1e-6
        
        adam = optimizers.Adam(learning_rate)
        sgd = optimizers.SGD(learning_rate)
        rmsprop = optimizers.RMSprop(learning_rate)
        adadelta = optimizers.Adadelta(learning_rate)
        adagrad = optimizers.Adagrad(learning_rate)
        adamax = optimizers.Adamax(learning_rate)

        model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), optimizer=rmsprop, metrics=["binary_accuracy"])
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
            model.summary()

            history = model.fit(chinese, 
            epochs=ep, validation_data=chinese_val, 
            callbacks=[early, lr_reduce],verbose=verbose_param)

            model.trainable = True


            K.set_value(model.optimizer.learning_rate, learning_rate_fine)
            #model.summary()
            
            rmsprop = optimizers.RMSprop(learning_rate_fine)
            model.compile(loss="binary_crossentropy", optimizer=rmsprop, metrics=["binary_accuracy"])

            history_fine = model.fit(chinese, 
            epochs=eps_fine, validation_data=chinese_val,
            callbacks=[early, lr_reduce],verbose=verbose_param)
            print('there')
            
            
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

            history = model.fit(french,  
            epochs=ep, validation_data=french_val, callbacks=[early, lr_reduce], verbose=verbose_param)

            model.trainable = True
            K.set_value(model.optimizer.learning_rate, learning_rate_fine)
            
            history_fine = model.fit(french, 
            epochs=eps_fine, validation_data=french_val,
            callbacks=[early, lr_reduce],verbose=verbose_param)
            
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
            epochs=eps_fine, validation_data=dataset_val, callbacks=[early, lr_reduce], verbose=verbose_param)

            model.trainable = True
            K.set_value(model.optimizer.learning_rate, learning_rate_fine)
            
            history_fine = model.fit(dataset, 
            epochs=ep, validation_data=dataset_val, 
            callbacks=[early, lr_reduce],verbose=verbose_param)
            
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


            train_acc = history_fine.history['accuracy']
            val_acc = history_fine.history['val_accuracy']
            train_loss = history_fine.history['loss']
            val_loss = history_fine.history['val_loss']
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
        model = Model(inputs=model.inputs, outputs=model.get_layer(name="feature_extractor").output)
        
        #model.summary()
        
        print('FEATURE EXTRACTION')
        features = []
        for i in CX:
            x = np.reshape(i, (itd.size,itd.size))
            x = cv2.merge([x,x,x])
            x = image.img_to_array(x)
            x = np.expand_dims(x, axis=0)
            feature = model.predict(x, verbose = 0)
            features.append(feature[0])
            
        self.CX = np.array(features)
        self.CY = CY

        features = []
        for i in CXT:
            x = np.reshape(i, (itd.size,itd.size))
            x = cv2.merge([x,x,x])
            x = image.img_to_array(x)
            x = np.expand_dims(x, axis=0)
            feature = model.predict(x, verbose = 0)
            features.append(feature[0])
        
        self.CXT = np.array(features)
        self.CYT = CYT

        features = []
        for i in FX:
            x = np.reshape(i, (itd.size,itd.size))
            x = cv2.merge([x,x,x])
            x = image.img_to_array(x)
            x = np.expand_dims(x, axis=0)
            feature = model.predict(x, verbose = 0)
            features.append(feature[0])
        
        self.FX = np.array(features)
        self.FY = FY

        features = []
        for i in FXT:
            x = np.reshape(i, (itd.size,itd.size))
            x = cv2.merge([x,x,x])
            x = image.img_to_array(x)
            x = np.expand_dims(x, axis=0)
            feature = model.predict(x, verbose = 0)
            features.append(feature[0])
        
        self.FXT = np.array(features)
        self.FYT = FYT

        features = []
        for i in MX:
            x = np.reshape(i, (itd.size,itd.size))
            x = cv2.merge([x,x,x])
            x = image.img_to_array(x)
            x = np.expand_dims(x, axis=0)
            feature = model.predict(x, verbose = 0)
            features.append(feature[0])
        
        self.MX = np.array(features)
        self.MY = MY

        features = []
        for i in MXT:
            x = np.reshape(i, (itd.size,itd.size))
            x = cv2.merge([x,x,x])
            x = image.img_to_array(x)
            x = np.expand_dims(x, axis=0)
            feature = model.predict(x, verbose = 0)
            features.append(feature[0])
        
        self.MXT = np.array(features)
        self.MYT = MYT

        

        







