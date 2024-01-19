import sys
sys.path.insert(1, '../../../')

from deep_learning_mitigation.classificator import ClassificatorClass
from deep_learning_mitigation.strings import Strings
from deep_learning_mitigation.ADVERSARY.test_robustness import TestRobustness
import numpy as np

strings = Strings()
paths = strings.carpet_paths_str
file_name = 'c_ind.csv'


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
        self.cc = ClassificatorClass(culture=culture,
                 greyscale=greyscale,
                 paths=paths,
                 times=1,
                 fileName=fileName,
                 validation_split=validation_split,
                 batch_size=batch_size,
                 epochs=epochs,
                 learning_rate=learning_rate,
                 verbose=verbose,
                 percent=percent,
                 plot=plot,
                 run_eagerly=run_eagerly,
                 lambda_index=lambda_index,
                 gpu=gpu)
        self.times = times
        self.cc.execute()
        self.test_rob = TestRobustness(
                            model=self.cc.model,
                            paths=paths,
                            culture=culture,
                            flat=greyscale,
                            percent=percent,
                            lambda_index=lambda_index,
                            lr=learning_rate,
                            epochs=epochs,
                            verbose_param=verbose,
                            fileName=file_name)
        
    def execute(self, n):
        g_rots = np.logspace(-4, 0, n)
        epss = np.logspace(-3, 0, n**2)
        for i in range(1,n**2):
            self.test_rob.robds.TestS = self.cc.TestSet
            self.cc.resetTestSet()
            self.test_rob.model = self.cc.model
            self.test_rob.test_on_augmented(g_rots[int(i/n)],g_rots[i%n], g_rots[i%n])
            self.test_rob.test_on_FGMA(epss[i])
            #self.test_rob.test_on_PGDA(epss[i])
        for j in range(1,self.times):
            self.cc.execute()
            for i in range(1,n**2):
                self.test_rob.robds.TestS = self.cc.TestSet
                self.cc.resetTestSet()
                self.test_rob.model = self.cc.model
                self.test_rob.test_on_augmented(g_rots[int(i/n)],g_rots[i%n], g_rots[i%n])
                self.test_rob.test_on_FGMA(epss[i])
                #self.test_rob.test_on_PGDA(epss[i])
        del self.cc
        del self.test_rob
        del g_rots
        del epss