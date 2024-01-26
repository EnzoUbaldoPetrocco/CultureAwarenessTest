import sys
sys.path.insert(1, '../../../')

from deep_learning_mitigation.classificator import ClassificatorClass
from deep_learning_mitigation.strings import Strings
from deep_learning_mitigation.ADVERSARY.test_robustness import TestRobustness
import numpy as np
import gc

strings = Strings()
paths = strings.carpet_paths_str


class Midware():
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
                 gpu = True):
        self.culture = culture
        self.greyscale = greyscale
        self.paths = paths
        self.fileName = fileName
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.percent = percent
        self.plot = plot
        self.run_eagerly = run_eagerly
        self.lambda_index = lambda_index
        self.gpu = gpu
        self.times = times
        
        
    def execute(self, n):
        g_rots = np.logspace(-4, 0, n)
        epss = np.logspace(-3, 0, n**2)
        cc = ClassificatorClass(culture=self.culture,
                 greyscale=self.greyscale,
                 paths=paths,
                 times=1,
                 fileName=self.fileName,
                 validation_split=self.validation_split,
                 batch_size=self.batch_size,
                 epochs=self.epochs,
                 learning_rate=self.learning_rate,
                 verbose=self.verbose,
                 percent=self.percent,
                 plot=self.plot,
                 run_eagerly=self.run_eagerly,
                 lambda_index=self.lambda_index,
                 gpu=self.gpu)
        cc.execute()
        test_rob = TestRobustness(
                            model=cc.model,
                            paths=self.paths,
                            culture=self.culture,
                            flat=self.greyscale,
                            percent=self.percent,
                            lambda_index=self.lambda_index,
                            lr=self.learning_rate,
                            epochs=self.epochs,
                            verbose_param=self.verbose,
                            fileName=self.fileName)
        for i in range(1,n**2):
            test_rob.robds.TestS = cc.TestSet
            cc.resetTestSet()
            test_rob.model = cc.model
            test_rob.test_on_augmented(g_rots[int(i/n)],g_rots[i%n], g_rots[i%n])
            test_rob.test_on_FGMA(epss[i])
            #self.test_rob.test_on_PGDA(epss[i])
        cc = None
        test_rob = None
        del cc
        del test_rob
        gc.collect()

        for j in range(1,self.times):
            cc = ClassificatorClass(culture=self.culture,
                 greyscale=self.greyscale,
                 paths=paths,
                 times=1,
                 fileName=self.fileName,
                 validation_split=self.validation_split,
                 batch_size=self.batch_size,
                 epochs=self.epochs,
                 learning_rate=self.learning_rate,
                 verbose=self.verbose,
                 percent=self.percent,
                 plot=self.plot,
                 run_eagerly=self.run_eagerly,
                 lambda_index=self.lambda_index,
                 gpu=self.gpu)
            cc.execute()
            test_rob = TestRobustness(
                                model=cc.model,
                                paths=self.paths,
                                culture=self.culture,
                                flat=self.greyscale,
                                percent=self.percent,
                                lambda_index=self.lambda_index,
                                lr=self.learning_rate,
                                epochs=self.epochs,
                                verbose_param=self.verbose,
                                fileName=self.fileName)
            for i in range(1,n**2):
                test_rob.robds.TestS = cc.TestSet
                cc.resetTestSet()
                test_rob.model = cc.model
                test_rob.test_on_augmented(g_rots[int(i/n)],g_rots[i%n], g_rots[i%n])
                test_rob.test_on_FGMA(epss[i])
                #self.test_rob.test_on_PGDA(epss[i])
            cc = None
            test_rob = None
            del cc
            del test_rob
            gc.collect()
