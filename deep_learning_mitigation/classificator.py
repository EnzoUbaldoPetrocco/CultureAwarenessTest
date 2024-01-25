import sys

sys.path.insert(1, '../')
import DS.ds
from Utils.utils import FileClass, ResultsClass
import numpy as np
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from keras.applications.resnet import ResNet50
from keras import layers, optimizers
import tensorflow as tf
from keras.layers import Dense, Flatten, Input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from matplotlib import pyplot as plt
import keras.backend as K
from keras.models import Model
import random
import gc
import os
import time
from cleverhans.tf2.utils import optimize_linear


class ClassificatorClass:
    def __init__(self,
                 culture=0,
                 greyscale=0,
                 paths=None,
                 times=30,
                 fileName='results.csv',
                 validation_split=0.1,
                 batch_size=1,
                 epochs=10,
                 learning_rate=1e-3,
                 verbose=0,
                 percent=0.1,
                 plot = False,
                 run_eagerly = False,
                 lambda_index = 0,
                 gpu = True,
                 sv_model = False):
        self.culture = culture
        self.greyscale = greyscale
        self.paths = paths
        self.times = times
        self.fileName = fileName
        self.resultsObj = ResultsClass()
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.verbose_param = verbose
        self.percent = percent
        self.plot = plot
        self.run_eagerly = run_eagerly
        self.lambda_index = lambda_index
        self.gpu = gpu
        self.sv_model = sv_model
        if self.gpu:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
            # Restrict TensorFlow to only allocate 2GB of memory on the first GPU
                try:
                    tf.config.experimental.set_virtual_device_configuration(
                        gpus[0],
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2600)])
                    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                    #tf.config.experimental.set_memory_growth(gpus[0], True)
                    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
                except RuntimeError as e:
                    # Virtual devices must be set before GPUs have been initialized
                    print(e)
            else:
                print('no gpus')
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        #lambda_grid = [1.00000000e-02, 1.46779927e-02, 2.15443469e-02,  3.16227766e-02,
        #4.64158883e-02, 6.81292069e-02, 1.00000000e-01, 1.46779927e-01,
        #2.15443469e-01, 3.16227766e-01, 4.64158883e-01, 6.81292069e-01,
        #1.00000000e+00, 1.46779927e+00, 2.15443469e+00, 3.16227766e+00,
        #4.64158883e+00, 6.81292069e+00, 1.00000000e+01, 1.46779927e+01,
        #2.15443469e+01, 3.16227766e+01, 4.64158883e+01, 6.81292069e+01,
        #1.00000000e+02]
        if lambda_index>=0:
            lambda_grid = np.logspace(-3,2,31)
            self.lamb = lambda_grid[lambda_index]
        else:
            self.lamb = 0

    def custom_loss(self, out):
        def loss(y_true, y_pred):
            weights1 = self.model.layers[len(self.model.layers)-1].kernel
            weights2 = self.model.layers[len(self.model.layers)-2].kernel
            weights3 = self.model.layers[len(self.model.layers)-3].kernel
            mean = tf.math.add(weights1,weights2)
            mean = tf.math.add(mean,weights3)
            mean = tf.multiply(mean,1/3)
            mean = tf.multiply(mean,self.lamb)
            if out == 0:
                dist = tf.norm(weights1-mean,ord='euclidean')
            if out == 1:
                dist = tf.norm(weights2-mean,ord='euclidean')
            if out == 2:
                dist = tf.norm(weights3-mean,ord='euclidean')
            dist = tf.multiply(dist,dist)
            #dist12 = tf.norm(weights1-weights2, ord='euclidean')
            #dist13 = tf.norm(weights1-weights3, ord='euclidean')
            #dist23 = tf.norm(weights2-weights3, ord='euclidean')
            #dist = tf.math.add(dist12, dist13)
            #dist = tf.math.add(dist, dist23)
            #dist = tf.multiply(tf.multiply(dist,dist) , self.lamb)
            loss = tf.keras.losses.binary_crossentropy(y_true[0][1], y_pred[0])
            res = tf.math.add(loss , dist)
            mask = tf.reduce_all(tf.equal(y_true[0][0], out))
            
            if not mask:
                return 0.0
            else:
                return res
        return loss

    def prepareDataset(self, paths):
        datasetClass = DS.ds.DSClass()
        datasetClass.mitigation_dataset(paths)
        self.TS = datasetClass.TS
        self.TestSet = datasetClass.TestS

    def quantize(self, yF):
        values = []
        for y in yF:
            if y>0.5:
                values.append(1)
            else:
                values.append(0)
        return values

    def test(self, testSet):
        testSet = np.array(testSet, dtype=object)
        XT = list(testSet[:, 0])
        yT = list(testSet[:, 1])
        XT = tf.stack(XT)
        yT = tf.stack(yT)
        yT = yT[:, 1]
        
        yFs = self.model.predict(XT)
        yFs = np.array(yFs, dtype=object)
        yFs = yFs[:,:,0]
        cms = []
        
        for yF in yFs:
            yF = self.quantize(yF)
            cm = confusion_matrix(yT, yF)
            cms.append(cm)        
        return cms

    def save_cm(self, fileName, cm):
        f = FileClass(fileName)
        f.writecm(cm)

    def get_results(self, fileName):
        f = FileClass(fileName)
        return f.readcms()

    def plot_training(self):
        if self.culture == 0:
            dense_acc_str = 'dense_accuracy'
            val_dense_acc_str = 'val_dense_accuracy'
            dense_loss_str= 'dense_loss'
            val_dense_loss_str = 'val_dense_loss'
        else:
            dense_acc_str = f'dense_{self.culture}_accuracy'
            val_dense_acc_str = f'val_dense_{self.culture}_accuracy'
            dense_loss_str = f'dense_{self.culture}_loss'
            val_dense_loss_str = f'val_dense_{self.culture}_loss'

        train_acc = self.history.history[dense_acc_str]
        val_acc = self.history.history[val_dense_acc_str]
        train_loss = self.history.history[dense_loss_str]
        val_loss = self.history.history[val_dense_loss_str]
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
        plt.plot(val_loss_x, val_loss, marker = '.', color = 'red', markersize = 10, 
                        linewidth = 1.5, label = 'Validation Loss')
        plt.title('Training Loss and Testing Loss w.r.t Number of Epochs')
        plt.legend()
        plt.show()

    def train(self, TS):
        size = np.shape(TS[0][0])
        input = Input(size)
        x = tf.keras.Sequential([
            ResNet50(input_shape=size, weights='imagenet', include_top=False)
        ])(input)
        x = Flatten()(x)
        y1 = (Dense(1, activation='sigmoid', name='dense'))(x)
        y2 = (Dense(1, activation='sigmoid', name='dense_1'))(x)
        y3 = (Dense(1, activation='sigmoid', name='dense_2'))(x)
        self.model = Model(inputs=input,
                    outputs = [y1,y2,y3],
                    name = 'model')
        self.model.trainable = True
        for layer in self.model.layers[1].layers:
            layer.trainable = False
        for layer in self.model.layers[1].layers[-3:]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True
        if self.culture == 0:
            monitor_val = 'val_dense_accuracy'
        else:
            monitor_val = f'val_dense_{self.culture}_accuracy'
        lr_reduce = ReduceLROnPlateau(monitor=monitor_val,
                                      factor=0.1,
                                      patience=int(self.epochs/3) + 1,
                                      verbose=self.verbose_param,
                                      mode='max',
                                      min_lr=1e-8)
        early = EarlyStopping(monitor=monitor_val,
                              min_delta=0.001,
                              patience=int(self.epochs/1.7) + 1,
                              verbose=self.verbose_param,
                              mode='auto')
        adam = optimizers.Adam(self.learning_rate)
        optimizer = adam
        
        self.model.compile(optimizer=optimizer,
                      metrics=["accuracy"],
                      loss=[self.custom_loss(out=0),self.custom_loss(out=1),self.custom_loss(out=2)], run_eagerly=self.run_eagerly)

        TS = np.array(TS, dtype=object)
        random.shuffle(TS)
        X = list(TS[:, 0])
        y = list(TS[:, 1])
        del TS
        X_val = X[int(len(X) * (1-self.validation_split)):len(X) - 1]
        y_val = y[int(len(y) * (1-self.validation_split)):len(y) - 1]
        X = X[0:int(len(X) * (1-self.validation_split))]
        y = y[0:int(len(y) * (1-self.validation_split))]
        
        X = tf.stack(X)
        y = tf.stack(y)
        X_val = tf.stack(X_val)
        y_val = tf.stack(y_val)
        tf.get_logger().setLevel('ERROR')
        #print(np.linalg.norm(np.array([i[0] for i in self.model.layers[len(self.model.layers)-2].get_weights()])-np.array([i[0] for i in self.model.layers[len(self.model.layers)-1].get_weights()])))
        self.history = self.model.fit(X,y,
                                 epochs=self.epochs,
                                 validation_data=(X_val,y_val),
                                 callbacks=[early, lr_reduce],
                                 verbose=self.verbose_param,
                                 batch_size = self.batch_size)
        #print(np.linalg.norm(np.array([i[0] for i in self.model.layers[len(self.model.layers)-2].get_weights()])-np.array([i[0] for i in self.model.layers[len(self.model.layers)-1].get_weights()])))
            
        if self.plot:
            self.plot_training()
        if self.sv_model:
            self.save_models()
    
    def augment_ds(self, lv_set):
        augmented_dataset = []
        data_augmentation = tf.keras.Sequential([
                 tf.keras.layers.RandomFlip("horizontal_and_vertical"),
                 tf.keras.layers.RandomRotation(self.g_rot),
                 tf.keras.layers.GaussianNoise(self.g_noise),
                 tf.keras.layers.RandomBrightness(self.g_bright)
            ])
        for culture in range(3):
            cultureTS = []
            for X, y in lv_set:
                X_augmented = data_augmentation(X, training=True)
                cultureTS.append((X_augmented, y))
        X = tf.stack(augmented_dataset)
        return X
    
    @tf.function
    def my_compute_gradient(self, loss_fn, x, y, targeted, culture=0):
        with tf.GradientTape() as g:
            g.watch(x)
            # Compute loss
            loss = loss_fn(y, self.model(x)[culture])
            if (
                targeted
            ):  # attack is targeted, minimize loss of target label rather than maximize loss of correct label
                loss = -loss

        # Define gradient of loss wrt input
        grad = g.gradient(loss, x)
        return grad
    
    def my_fast_gradient_method(
        self,
        x,
        eps,
        norm,
        loss_fn=None,
        clip_min=None,
        clip_max=None,
        y=None,
        targeted=False,
        sanity_checks=False,
        culture=0,
        plot = None
    ):
        if norm not in [np.inf, 1, 2]:
            raise ValueError("Norm order must be either np.inf, 1, or 2.")

        if loss_fn is None:
            loss_fn = tf.nn.sparse_softmax_cross_entropy_with_logits

        asserts = []

        # If a data range was specified, check that the input was in that range
        if clip_min is not None:
            asserts.append(tf.math.greater_equal(x, clip_min))

        if clip_max is not None:
            asserts.append(tf.math.less_equal(x, clip_max))

        # cast to tensor if provided as numpy array
        x = tf.cast(x, tf.float32)

        if y is None:
            # Using model predictions as ground truth to avoid label leaking
            y = tf.argmax(self.model(x)[culture], 1)

        grad = self.my_compute_gradient( loss_fn, x, y, targeted, culture=culture)

        optimal_perturbation = optimize_linear(grad, eps, norm)

        if plot is not None:
            plt.imshow(optimal_perturbation[0])
            plt.show()

        # Add perturbation to original example to obtain adversarial example
        adv_x = x + optimal_perturbation

        # If clipping is needed, reset all values outside of [clip_min, clip_max]
        if (clip_min is not None) or (clip_max is not None):
            # We don't currently support one-sided clipping
            assert clip_min is not None and clip_max is not None
            adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)

        if sanity_checks:
            assert np.all(asserts)
        return adv_x

    def fast_gradient_method_augmentation(self, lv_set, eps=0.3):
        augmented_dataset = []
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        for X, y in lv_set:
            X = X[None, ...]
            y_i = y[1]*2-1
            y_i = tf.constant(y_i, dtype=tf.int32)
            y_i = y_i[None, ...]
            X_augmented = self.my_fast_gradient_method(X,
                 eps, np.inf, y=y_i, culture=y[0], loss_fn=bce)
            #print(f'X=Xaugmented = {X==X_augmented}')
            #f, axarr = plt.subplots(2,1)
            #axarr[0].imshow(X[0][:, :, ::-1])
            #axarr[1].imshow(X_augmented[0][:, :, ::-1])
            #plt.show()

            augmented_dataset.append((X_augmented[0], y))

    def train_augmented(self, TS):
        size = np.shape(TS[0][0])
        input = Input(size)
        x = tf.keras.Sequential([
            ResNet50(input_shape=size, weights='imagenet', include_top=False)
        ])(input)
        x = Flatten()(x)
        y1 = (Dense(1, activation='sigmoid', name='dense'))(x)
        y2 = (Dense(1, activation='sigmoid', name='dense_1'))(x)
        y3 = (Dense(1, activation='sigmoid', name='dense_2'))(x)
        self.model = Model(inputs=input,
                    outputs = [y1,y2,y3],
                    name = 'model')
        self.model.trainable = True
        for layer in self.model.layers[1].layers:
            layer.trainable = False
        for layer in self.model.layers[1].layers[-3:]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True
        if self.culture == 0:
            monitor_val = 'val_dense_accuracy'
        else:
            monitor_val = f'val_dense_{self.culture}_accuracy'
        lr_reduce = ReduceLROnPlateau(monitor=monitor_val,
                                      factor=0.1,
                                      patience=int(self.epochs/3) + 1,
                                      verbose=self.verbose_param,
                                      mode='max',
                                      min_lr=1e-8)
        early = EarlyStopping(monitor=monitor_val,
                              min_delta=0.001,
                              patience=int(self.epochs/1.7) + 1,
                              verbose=self.verbose_param,
                              mode='auto')
        adam = optimizers.Adam(self.learning_rate)
        optimizer = adam
        
        self.model.compile(optimizer=optimizer,
                      metrics=["accuracy"],
                      loss=[self.custom_loss(out=0),self.custom_loss(out=1),self.custom_loss(out=2)], run_eagerly=self.run_eagerly)

        TS = np.array(TS, dtype=object)

        ###################################################
        # AUGMENT DATASET #################################
        augmented_dataset = self.augment_ds(TS)
        ###################################################
        TS = np.append(TS, augmented_dataset, axis=0)

        random.shuffle(TS)
        X = list(TS[:, 0])
        y = list(TS[:, 1])
        del TS
        del augmented_dataset
        X_val = X[int(len(X) * (1-self.validation_split)):len(X) - 1]
        y_val = y[int(len(y) * (1-self.validation_split)):len(y) - 1]
        X = X[0:int(len(X) * (1-self.validation_split))]
        y = y[0:int(len(y) * (1-self.validation_split))]
        
        X = tf.stack(X)
        y = tf.stack(y)
        

        X_val = tf.stack(X_val)
        y_val = tf.stack(y_val)
        tf.get_logger().setLevel('ERROR')
        #print(np.linalg.norm(np.array([i[0] for i in self.model.layers[len(self.model.layers)-2].get_weights()])-np.array([i[0] for i in self.model.layers[len(self.model.layers)-1].get_weights()])))
        self.history = self.model.fit(X,y,
                                 epochs=self.epochs,
                                 validation_data=(X_val,y_val),
                                 callbacks=[early, lr_reduce],
                                 verbose=self.verbose_param,
                                 batch_size = self.batch_size)
        #print(np.linalg.norm(np.array([i[0] for i in self.model.layers[len(self.model.layers)-2].get_weights()])-np.array([i[0] for i in self.model.layers[len(self.model.layers)-1].get_weights()])))
            
        if self.plot:
            self.plot_training()
        if self.sv_model:
            self.save_models()

    def execute(self):
        for i in range(self.times):
            gc.collect()
            print(f'CICLE {i}')
            obj = DS.ds.DSClass()
            obj.mitigation_dataset(self.paths, self.greyscale, 0)
            obj.nineonedivision(self.culture, percent=self.percent)
            # I have to select a culture
            TS = obj.TS[self.culture]
            # I have to test on every culture
            TestSets = obj.TestS
            # Name of the file management for results
            fileNames = []
            for l in range(len(TestSets)):
                onPointSplitted = self.fileName.split('.')
                fileNamesOut = []
                for o in range(3):
                    name = 'percent' + str(self.percent).replace('.', ',') + '/' +  str(self.lambda_index) + '/' + 'normal_test/' +  onPointSplitted[0] + str(
                        l) + f'/out{o}.' + onPointSplitted[1]
                    
                    fileNamesOut.append(name)
                    del name
                fileNames.append(fileNamesOut)
                del onPointSplitted
                
            self.train(TS)
            cms = []
            for k, TestSet in enumerate(TestSets):
                cm = self.test(self.model, TestSet)
                for o in range(3):
                    print(fileNames[k][o])
                    self.save_cm(fileNames[k][o], cm[o])
                    cms.append(cm)
            # Reset Memory each time
            self.TestSet = TestSets
            del TS
            del obj
            del TestSets
            gc.collect()
        
        if self.verbose_param:
            for i in range(3):
                for o in range(3):
                    result = self.get_results(fileNames[i][o])
                    result = np.array(result, dtype=object)
                    print(f'RESULTS OF CULTURE {i}, out {o}')
                    tot = self.resultsObj.return_tot_elements(result[0])
                    pcm_list = self.resultsObj.calculate_percentage_confusion_matrix(
                        result, tot)
                    statistic = self.resultsObj.return_statistics_pcm(pcm_list)
                    print(statistic[0])
                    accuracy = statistic[0][0][0] + statistic[0][1][1]
                    print(f'Accuracy is {accuracy} %')
    
    def resetTestSet(self):
        self.TestSets = None
        
    def execute_model_selection(self, bs= True):
        for i in range(self.times):
            gc.collect()
            print(f'CICLE {i}')
            obj = DS.ds.DSClass()
            obj.mitigation_dataset(self.paths, self.greyscale, 0)
            obj.nineonedivision(self.culture, percent=self.percent)
            # I have to select a culture
            TS = obj.TS[self.culture]
            # I have to test on every culture
            TestSets = obj.TestS
            # Name of the file management for results
            fileNames = []
            for l in range(len(TestSets)):
                onPointSplitted = self.fileName.split('.')
                fileNamesOut = []
                for o in range(3):
                    name = 'percent' + str(self.percent).replace('.', ',') + '/' +  str(self.lambda_index) + '/' + onPointSplitted[0] + str(
                        l) + f'/out{o}.' + onPointSplitted[1]
                    fileNamesOut.append(name)
                    del name
                fileNames.append(fileNamesOut)
                del onPointSplitted
                del fileNamesOut
            self.model_selection(TS, bs)
            cms = []
            for k, TestSet in enumerate(TestSets):
                cm = self.test(TestSet)
                for o in range(3):
                    print(fileNames[k][o])
                    self.save_cm(fileNames[k][o], cm[o])
                    cms.append(cm)
            # Reset Memory each time
            del TS
            del obj
            del TestSets
            gc.collect()
        
        if self.verbose_param:
            for i in range(len(obj.TS)):
                for o in range(3):
                    result = self.get_results(fileNames[i][o])
                    result = np.array(result, dtype=object)
                    print(f'RESULTS OF CULTURE {i}, out {o}')
                    tot = self.resultsObj.return_tot_elements(result[0])
                    pcm_list = self.resultsObj.calculate_percentage_confusion_matrix(
                        result, tot)
                    statistic = self.resultsObj.return_statistics_pcm(pcm_list)
                    print(statistic[0])
                    
                    accuracy = statistic[0][0][0] + statistic[0][1][1]
                    print(f'Accuracy is {accuracy} %')

    def save_models(self):
        culture_path = './' + self.fileName.split('.')[0]
        percent_path = culture_path + '/' + str(self.percent)
        model_path = percent_path + '/' + str(self.lambda_index)
        fileUtil = FileClass(self.fileName)
        fileUtil.mkdir(culture_path)
        fileUtil.mkdir(percent_path)
        fileUtil.mkdir(model_path)
        count = 0
        # Iterate directory
        for path in os.listdir(model_path):
            # check if current path is a file
            if os.path.isfile(os.path.join(model_path, path)):
                count += 1
        self.model.save_weights(f'{model_path}/checkpoint_{count}')

    def model_selection(self, TS, batch_size=True):
        size = np.shape(TS[0][0])
        input = Input(size)
        x = tf.keras.Sequential([
            ResNet50(input_shape=size, weights='imagenet', include_top=False)
        ])(input)
        x = Flatten()(x)
        y1 = (Dense(1, activation='sigmoid', name='dense'))(x)
        y2 = (Dense(1, activation='sigmoid', name='dense_1'))(x)
        y3 = (Dense(1, activation='sigmoid', name='dense_2'))(x)
        if batch_size:
            bs_list = [2] # batch size list
        else:
            bs_list = [self.batch_size]
        lr_list = np.logspace(-7.5,-2.5,18)
        act_val_acc = 0
        for bs in bs_list:
            for lr in lr_list:
                if self.verbose_param:
                    print(f'For batch size = {bs} and learning rate = {lr}')
                self.model = Model(inputs=input,
                    outputs = [y1,y2,y3],
                    name = 'model')
                self.model.trainable = True
                for layer in self.model.layers[1].layers:
                    layer.trainable = False
                for layer in self.model.layers[1].layers[-3:]:
                    if not isinstance(layer, layers.BatchNormalization):
                        layer.trainable = True
                if self.culture == 0:
                    monitor_val = 'val_dense_accuracy'
                else:
                    monitor_val = f'val_dense_{self.culture}_accuracy'
                lr_reduce = ReduceLROnPlateau(monitor=monitor_val,
                                      factor=0.1,
                                      patience=int(self.epochs/3) + 1,
                                      verbose=self.verbose_param,
                                      mode='max',
                                      min_lr=1e-8)
                early = EarlyStopping(monitor=monitor_val,
                                    min_delta=0.001,
                                    patience=int(self.epochs/1.7) + 1,
                                    verbose=self.verbose_param,
                                    mode='auto')
                adam = optimizers.Adam(lr)
                optimizer = adam
                self.model.compile(optimizer=optimizer,
                      metrics=["accuracy"],
                      loss=[self.custom_loss(out=0),self.custom_loss(out=1),self.custom_loss(out=2)], run_eagerly=self.run_eagerly)

                TS = np.array(TS, dtype=object)
                X = list(TS[:, 0])
                y = list(TS[:, 1])
                X_val = X[int(len(X) * (1-self.validation_split)):len(X) - 1]
                y_val = y[int(len(y) * (1-self.validation_split)):len(y) - 1]
                X = X[0:int(len(X) * (1-self.validation_split))]
                y = y[0:int(len(y) * (1-self.validation_split))]
                X = tf.stack(X)
                y = tf.stack(y)
                X_val = tf.stack(X_val)
                y_val = tf.stack(y_val)
                self.history = self.model.fit(X,y,
                                        epochs=self.epochs,
                                        validation_data=(X_val,y_val),
                                        callbacks=[early, lr_reduce],
                                        verbose=self.verbose_param,
                                        batch_size = bs)
                val_acc = self.history.history[monitor_val]
                if act_val_acc < val_acc[-1]:
                    best_model = self.model
                    self.batch_size = bs
                    self.learning_rate = lr
                if self.plot:
                    self.plot_training()
                self.model = None
                time.sleep(5)
        print(f'best hyperparams are: lr: {self.learning_rate}, bs: {self.batch_size}')
        self.model = best_model
        best_model = None
        if self.sv_model:
            self.save_models()
    
    def exe_intellig_model_selection(self, bs= True):
        gc.collect()
        print(f'CICLE {0}')
        obj = DS.ds.DSClass()
        obj.mitigation_dataset(self.paths, self.greyscale, 0)
        obj.nineonedivision(self.culture, percent=self.percent)
        # I have to select a culture
        TS = obj.TS[self.culture]
        # I have to test on every culture
        TestSets = obj.TestS
        # Name of the file management for results
        fileNames = []
        for l in range(len(TestSets)):
            onPointSplitted = self.fileName.split('.')
            fileNamesOut = []
            for o in range(3):
                name = 'percent' + str(self.percent).replace('.', ',') + '/' +  str(self.lambda_index) + '/' + onPointSplitted[0] + str(
                    l) + f'/out{o}.' + onPointSplitted[1]
                
                fileNamesOut.append(name)
            fileNames.append(fileNamesOut)
        self.model_selection(TS, bs)
        cms = []
        for k, TestSet in enumerate(TestSets):
            cm = self.test(self.model, TestSet)
            for o in range(3):
                print(fileNames[k][o])
                self.save_cm(fileNames[k][o], cm[o])
                cms.append(cm)
        # Reset Memory each time
        
                
        gc.collect()
        if self.times > 1:
            for i in range(self.times-1):
                gc.collect()
                print(f'CICLE {i}')
                obj = DS.ds.DSClass()
                obj.mitigation_dataset(self.paths, self.greyscale, 0)
                obj.nineonedivision(self.culture, percent=self.percent)
                # I have to select a culture
                TS = obj.TS[self.culture]
                # I have to test on every culture
                TestSets = obj.TestS
                # Name of the file management for results
                fileNames = []
                for l in range(len(TestSets)):
                    onPointSplitted = self.fileName.split('.')
                    fileNamesOut = []
                    for o in range(3):
                        name = 'percent' + str(self.percent).replace('.', ',') + '/' +  str(self.lambda_index) + '/' + onPointSplitted[0] + str(
                            l) + f'/out{o}.' + onPointSplitted[1]
                        
                        fileNamesOut.append(name)
                    fileNames.append(fileNamesOut)
                self.train(TS)
                cms = []
                for k, TestSet in enumerate(TestSets):
                    cm = self.test(self.model, TestSet)
                    for o in range(3):
                        print(fileNames[k][o])
                        self.save_cm(fileNames[k][o], cm[o])
                        cms.append(cm)
                # Reset Memory each time
                del TS
                del obj
                del TestSets
                gc.collect()
    
        if self.verbose_param:
            for i in range(len(obj.TS)):
                for o in range(3):
                    result = self.get_results(fileNames[i][o])
                    result = np.array(result, dtype=object)
                    print(f'RESULTS OF CULTURE {i}, out {o}')
                    tot = self.resultsObj.return_tot_elements(result[0])
                    pcm_list = self.resultsObj.calculate_percentage_confusion_matrix(
                        result, tot)
                    statistic = self.resultsObj.return_statistics_pcm(pcm_list)
                    print(statistic[0])
                    
                    accuracy = statistic[0][0][0] + statistic[0][1][1]
                    print(f'Accuracy is {accuracy} %')