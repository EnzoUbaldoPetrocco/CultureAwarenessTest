import sys

sys.path.insert(1, "../../")
from easydict import EasyDict
import DS.ds
import numpy as np
import os
from Utils.utils import ResultsClass
import tensorflow as tf
from deep_learning_mitigation.ADVERSARY.mit_model import MitigationModel


class AdversaryClassificator:
    def __init__(
        self,
        culture=0,
        greyscale=0,
        paths=None,
        times=30,
        fileName="results.csv",
        validation_split=0.1,
        batch_size=1,
        epochs=10,
        learning_rate=1e-3,
        verbose=0,
        percent=0.1,
        plot=False,
        run_eagerly=False,
        lambda_index=0,
        gpu=True,
        eps=0.3,
        w_path="./",
    ):
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
        if lambda_index<0:
            self.lamb = 0
        else:
            lambda_grid = np.logspace(-3, 2, 31)
            self.lamb = lambda_grid[lambda_index]
            
        self.gpu = gpu
        self.eps = eps
        self.w_path = w_path
        if self.gpu:
            gpus = tf.config.experimental.list_physical_devices("GPU")
            if gpus:
                # Restrict TensorFlow to only allocate 2GB of memory on the first GPU
                try:
                    tf.config.experimental.set_virtual_device_configuration(
                        gpus[0],
                        [
                            tf.config.experimental.VirtualDeviceConfiguration(
                                memory_limit=2600
                            )
                        ],
                    )
                    logical_gpus = tf.config.experimental.list_logical_devices("GPU")
                    # tf.config.experimental.set_memory_growth(gpus[0], True)
                    print(
                        len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs"
                    )
                except RuntimeError as e:
                    # Virtual devices must be set before GPUs have been initialized
                    print(e)
            else:
                print("no gpus")
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        

    def unit_cicle(self, iteration, checkpoint_m=0):
        data = self.load_ds(
            paths=self.paths,
            greyscale=self.greyscale,
            culture=self.culture,
            percent=self.percent,
            batch=self.batch_size,
        )
        self.size = np.shape(np.asarray(data.train[0][0][0], dtype=object))
        # ind = 0
        # if iteration!=0:
        #    ind = iteration*2+1
        path_weights = (
            self.w_path
            + self.fileName.split(".")[0]
            + "/"
            + str(self.percent)
            + "/"
            + str(self.lambda_index)
            + "/checkpoint_"
            + str(checkpoint_m)
        )
        # for every cycle I must evaluate the model:
        #   - Without Robust Samples
        #   - With Gradient Approach
        #   - Classical Data Augmentation
        #   - Coupled method

        clas = MitigationModel(
            lr=self.learning_rate,
            lambda_index=self.lambda_index,
            bs=self.batch_size,
            nb_epochs=self.epochs,
            eps=self.eps,
            size=self.size,
            verbose=self.verbose_param,
            plot=self.plot,
            percent=self.percent,
            prev_weights=True,
            path_weights=path_weights,
        )
        fileName = (
            self.fileName.split(".")[0] + "/standard." + self.fileName.split(".")[1]
        )
        clas.test(data=data, fileName=fileName, culture=self.culture)

        """
        # With ADV
        avd_clas = MitigationModel(
            lr=self.learning_rate,
            lambda_index=self.lambda_index,
            bs=self.batch_size,
            nb_epochs=self.epochs,
            eps=self.eps,
            size=self.size,
            verbose=self.verbose_param,
            plot=self.plot,
            percent=self.percent,
            prev_weights=True,
            path_weights=path_weights,
        )
        avd_clas.fit(data, True, False)
        avd_clas.test(
            data,
            self.fileName.split(".")[0] + "/adversarial." + self.fileName.split(".")[1],
        )
        
        # With DA
        avd_clas = MitigationModel(
            lr=self.learning_rate,
            lambda_index=self.lambda_index,
            bs=self.batch_size,
            nb_epochs=self.epochs,
            eps=self.eps,
            size=self.size,
            verbose=self.verbose_param,
            plot=self.plot,
            percent=self.percent,
            prev_weights=True,
            path_weights=path_weights,
        )
        avd_clas.fit(data, False, True)
        clas.test(
            data,
            self.fileName.split(".")[0]
            + "/data_augmentation."
            + self.fileName.split(".")[1],
        )
        # With ADV
        avd_clas = MitigationModel(
            lr=self.learning_rate,
            lambda_index=self.lambda_index,
            bs=self.batch_size,
            nb_epochs=self.epochs,
            eps=self.eps,
            size=self.size,
            verbose=self.verbose_param,
            plot=self.plot,
            percent=self.percent,
            prev_weights=True,
            path_weights=path_weights,
        )
        avd_clas.fit(data, True, True)
        clas.test(
            data, self.fileName.split(".")[0] + "/both." + self.fileName.split(".")[1]
        )"""

    def execute(self):
        for i in range(self.times):
            print(f"Cycle {i}")
            self.unit_cicle(i, checkpoint_m=0)

    def model_selection(self):
        ...

    def i_model_selection(self):
        ...

    # LOAD DS
    def load_ds(self, paths, greyscale, culture, percent, batch):
        def split_list(lst, chunk_size):
            return list(zip(*[iter(lst)] * chunk_size))

        obj = DS.ds.DSClass()
        obj.mitigation_dataset(paths, greyscale, 0)
        obj.nineonedivision(culture, percent=percent)
        # I have to select a culture
        TS = obj.TS[culture]
        # I have to test on every culture
        TestSets = obj.TestS
        TS = split_list(TS, batch)

        # TestSets = list(np.asarray(TestSets,dtype=object)[:,0:10])
        # TS = list(np.asarray(TS,dtype=object)[0:10])
        return EasyDict(train=TS, test=TestSets)
