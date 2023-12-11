import sys

sys.path.insert(1, "../../../../")


from keras.applications.resnet import ResNet50
from keras import layers, optimizers
import tensorflow as tf
from keras.layers import Dense, Flatten, Input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import Model
from sklearn.metrics import confusion_matrix
import numpy as np
from Utils.utils import FileClass, ResultsClass
from deep_learning_mitigation.ADVERSARY.robust_ds import Robust_ds
from matplotlib import pyplot as plt


class TestRobustness:
    def __init__(
        self,
        model_path="",
        paths=None,
        greyscale=0,
        culture=0,
        flat=0,
        percent=0.1,
        size=(100, 100, 3),
        lambda_index=0,
        lr = 1e-3,
        epochs = 10, 
        verbose_param = 0,
        fileName = 'model.csv'
    ):
        self.size = size
        self.model_path = model_path
        self.learning_rate = lr
        self.epochs = epochs
        self.verbose_param = verbose_param
        self.culture = culture
        self.percent = percent
        self.ld_model(model_path)
        self.robds = Robust_ds(paths, greyscale, culture, flat, percent, self.model)
        self.fileName = fileName
        self.lambda_index = lambda_index

        if lambda_index>=0:
            lambda_grid = np.logspace(-3,2,31)
            self.lamb = lambda_grid[lambda_index]
        else:
            self.lamb = 0




    def ld_model(self, model_path):
        def custom_loss(out):
            def loss(y_true, y_pred):
                weights1 = model.layers[len(model.layers) - 1].kernel
                weights2 = model.layers[len(model.layers) - 2].kernel
                weights3 = model.layers[len(model.layers) - 3].kernel
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
                # dist12 = tf.norm(weights1-weights2, ord='euclidean')
                # dist13 = tf.norm(weights1-weights3, ord='euclidean')
                # dist23 = tf.norm(weights2-weights3, ord='euclidean')
                # dist = tf.math.add(dist12, dist13)
                # dist = tf.math.add(dist, dist23)
                # dist = tf.multiply(tf.multiply(dist,dist) , .lamb)
                loss = tf.keras.losses.binary_crossentropy(y_true[0][1], y_pred[0])
                res = tf.math.add(loss, dist)
                mask = tf.reduce_all(tf.equal(y_true[0][0], out))
                if not mask:
                    return 0.0
                else:
                    return res

            return loss

        input = Input(self.size)
        x = tf.keras.Sequential(
            [ResNet50(input_shape=self.size, weights="imagenet", include_top=False)]
        )(input)
        x = Flatten()(x)
        y1 = (Dense(1, activation="sigmoid", name="dense"))(x)
        y2 = (Dense(1, activation="sigmoid", name="dense_1"))(x)
        y3 = (Dense(1, activation="sigmoid", name="dense_2"))(x)
        model = Model(inputs=input, outputs=[y1, y2, y3], name="model")
        model.trainable = True
        for layer in model.layers[1].layers:
            layer.trainable = False
        for layer in model.layers[1].layers[-3:]:
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

        model.compile(
            optimizer=optimizer,
            metrics=["accuracy"],
            loss=[custom_loss(out=0), custom_loss(out=1), custom_loss(out=2)],
        )

        model.load_weights(model_path)
        self.model = model
        model = None

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
        yT = np.asarray(yT, dtype=float)
        yT = yT[:, 1]
        
        yT = self.quantize(yT)
        
        yT = tf.stack(yT)


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

    def get_filenames(self, len_TestSets, prefix):
        fileNames = []
        for l in range(len_TestSets):
                        onPointSplitted = self.fileName.split('.')
                        fileNamesOut = []
                        for o in range(3):
                            name = prefix +'percent' + str(self.percent).replace('.', ',') + '/' +  str(self.lambda_index) + '/' + onPointSplitted[0] + str(
                                l) + f'/out{o}.' + onPointSplitted[1]
                            
                            fileNamesOut.append(name)
                        fileNames.append(fileNamesOut)
        return fileNames

    def test_on_augmented(self, g_rot=0.2, g_noise=0.1):
        self.robds.standard_augmentation(g_rot=g_rot, g_noise=g_noise)
        cms = []
        TestSets = self.robds.augmented_dataset
        prefix = 'augmented/g_rot=' + str(g_rot) + '/g_noise=' + str(g_noise) + '/'
        fileNames = self.get_filenames(len(TestSets), prefix)
        for culture in range(len(TestSets)):
            cm = self.test(TestSets[culture])
            for o in range(3):
                    print(fileNames[culture][o])
                    self.save_cm(fileNames[culture][o], cm[o])
                    cms.append(cm)


    def test_on_FGMA(self, eps = 0.3):
        self.robds.fast_gradient_method_augmentation(eps=eps)
        cms = []
        TestSets = self.robds.fast_gradient_method_augmented_dataset
        prefix = 'fgma/eps=' + str(eps) + '/'
        fileNames = self.get_filenames(len(TestSets), prefix)

        for culture in range(len(TestSets)):
            cm = self.test(TestSets[culture])
            for o in range(3):
                    print(fileNames[culture][o])
                    self.save_cm(fileNames[culture][o], cm[o])
                    cms.append(cm)


    def test_on_PGDA(self, eps = 0.3):
        self.robds.projected_gradient_descent_augmentation(eps=eps)
        cms = []
        TestSets = self.robds.projected_gradient_decent_augmented_dataset
        prefix = 'fgma/pgda=' + str(eps) + '/'
        fileNames = self.get_filenames(len(TestSets), prefix)
        for culture in range(len(TestSets)):
            cm = self.test(TestSets[culture])
            for o in range(3):
                    print(fileNames[culture][o])
                    self.save_cm(fileNames[culture][o], cm[o])
                    cms.append(cm)


        