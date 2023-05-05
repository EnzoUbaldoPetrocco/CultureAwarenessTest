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
                 plot = False,
                 run_eagerly = False,
                 lambda_index = 0):
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
        self.plot = plot
        self.run_eagerly = run_eagerly
        self.lambda_index = lambda_index

        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
        # Restrict TensorFlow to only allocate 2GB of memory on the first GPU
            try:
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2300)])
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Virtual devices must be set before GPUs have been initialized
                print(e)
        else:
            print('no gpus')

        lambda_grid = [1.00000000e-02, 1.46779927e-02, 2.15443469e-02,  3.16227766e-02,
        4.64158883e-02, 6.81292069e-02, 1.00000000e-01, 1.46779927e-01,
        2.15443469e-01, 3.16227766e-01, 4.64158883e-01, 6.81292069e-01,
        1.00000000e+00, 1.46779927e+00, 2.15443469e+00, 3.16227766e+00,
        4.64158883e+00, 6.81292069e+00, 1.00000000e+01, 1.46779927e+01,
        2.15443469e+01, 3.16227766e+01, 4.64158883e+01, 6.81292069e+01,
        1.00000000e+02]
        self.lamb = lambda_grid[lambda_index]

    def custom_loss(self, out):
        def loss(y_true, y_pred):
            weights1 = self.model.layers[len(self.model.layers)-1].kernel
            weights2 = self.model.layers[len(self.model.layers)-2].kernel
            weights3 = self.model.layers[len(self.model.layers)-3].kernel
            dist12 = tf.norm(weights1-weights2, ord='euclidean')
            dist13 = tf.norm(weights1-weights3, ord='euclidean')
            dist23 = tf.norm(weights2-weights3, ord='euclidean')
            dist = tf.math.add(dist12, dist13)
            dist = tf.math.add(dist, dist23)
            dist = tf.multiply(tf.multiply(dist,dist) , self.lamb)
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

    def test(self, model, testSet):
        testSet = np.array(testSet, dtype=object)
        XT = list(testSet[:, 0])
        yT = list(testSet[:, 1])
        XT = tf.stack(XT)
        yT = tf.stack(yT)
        yT = yT[:, 1]
        
        yFs = model.predict(XT)
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
        train_acc = self.history.history['dense_accuracy']
        val_acc = self.history.history['val_dense_accuracy']
        train_loss = self.history.history['dense_loss']
        val_loss = self.history.history['val_dense_loss']
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
                                      factor=0.2,
                                      patience=3,
                                      verbose=self.verbose_param,
                                      mode='max',
                                      min_lr=1e-8)
        early = EarlyStopping(monitor=monitor_val,
                              min_delta=0.001,
                              patience=8,
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
        X_val = X[int(len(X) * (1-self.validation_split)):len(X) - 1]
        y_val = y[int(len(y) * (1-self.validation_split)):len(y) - 1]
        X = X[0:int(len(X) * (1-self.validation_split))]
        y = y[0:int(len(y) * (1-self.validation_split))]
        
        X = tf.stack(X)
        y = tf.stack(y)
        X_val = tf.stack(X_val)
        y_val = tf.stack(y_val)
        tf.get_logger().setLevel('ERROR')
        print(np.linalg.norm(np.array([i[0] for i in self.model.layers[len(self.model.layers)-2].get_weights()])-np.array([i[0] for i in self.model.layers[len(self.model.layers)-1].get_weights()])))
        self.history = self.model.fit(X,y,
                                 epochs=self.epochs,
                                 validation_data=(X_val,y_val),
                                 callbacks=[early, lr_reduce],
                                 verbose=self.verbose_param,
                                 batch_size = self.batch_size)
        print(np.linalg.norm(np.array([i[0] for i in self.model.layers[len(self.model.layers)-2].get_weights()])-np.array([i[0] for i in self.model.layers[len(self.model.layers)-1].get_weights()])))
            
        if self.plot:
            self.plot_training()
        return self.model

    def execute(self):
        for i in range(self.times):
            print(f'CICLE {i}')
            obj = DS.ds.DSClass()
            obj.mitigation_dataset(self.paths, self.greyscale, 0)
            obj.nineonedivision(self.culture)
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
                    name = str(self.lambda_index) + '/' + onPointSplitted[0] + str(
                        l) + f'/out{o}.' + onPointSplitted[1]
                    
                    fileNamesOut.append(name)
                fileNames.append(fileNamesOut)
            model = self.train(TS)
            cms = []
            for k, TestSet in enumerate(TestSets):
                cm = self.test(model, TestSet)
                for o in range(3):
                    print(fileNames[k][o])
                    self.save_cm(fileNames[k][o], cm[o])
                    cms.append(cm)
        if self.verbose:
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


