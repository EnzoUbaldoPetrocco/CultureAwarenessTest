import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
sys.path.insert(1, '../')
import DS.ds
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from Utils.utils import FileClass, ResultsClass
import gc
from minimization_regularized.regularized import Model
import time

class ClassificatorClass:
    def __init__(self, culture=0, greyscale=0, paths=None,
                 type='SVC', points=30, kernel='linear', times=30, fileName = 'results.csv',
                 validation_split = 0.25 ,learning_rate = 0.001, epochs = 800, lambda_index= 0, lamb = 0, verbose_param = 0 ):
        self.culture = culture
        self.greyscale = greyscale
        self.paths = paths
        self.type = type
        self.points = points
        self.kernel = kernel
        self.times = times
        self.fileName = fileName
        self.resultsObj = ResultsClass()
        self.validation_split = validation_split
        self.lr = learning_rate
        self.epochs = epochs
        self.lambda_index = lambda_index
        if lambda_index >= 0:
            lambda_grid = [1.00000000e-02, 1.46779927e-02, 2.15443469e-02,  3.16227766e-02,
            4.64158883e-02, 6.81292069e-02, 1.00000000e-01, 1.46779927e-01,
            2.15443469e-01, 3.16227766e-01, 4.64158883e-01, 6.81292069e-01,
            1.00000000e+00, 1.46779927e+00, 2.15443469e+00, 3.16227766e+00,
            4.64158883e+00, 6.81292069e+00, 1.00000000e+01, 1.46779927e+01,
            2.15443469e+01, 3.16227766e+01, 4.64158883e+01, 6.81292069e+01,
            1.00000000e+02]
            self.lamb = lambda_grid[lambda_index]
        else: 
            self.lamb = lamb
        self.verbose_param = verbose_param
        
    def prepareDataset(self, paths):
        datasetClass = DS.ds.DSClass()
        datasetClass.build_dataset(paths)
        self.TS = datasetClass.TS
        self.TestSet = datasetClass.TestS

    def SVC(self, TS):
        if self.kernel == 'rbf':
            print('RBF IS NOT IMPLEMENTED')
            logspaceC = np.logspace(-2,2,self.points)
            logspaceGamma = np.logspace(-2,2,self.points)
        if self.kernel == 'linear':
            logspaceC = np.logspace(-2,2,self.points)
            logspaceGamma = np.logspace(-2,2,self.points)

        # training set is divided into (X,y)
        TS = np.array(TS, dtype = object)
        X = list(TS[:,0])
        y = list(TS[:,1])
        trainY = []
        for y_i in y:
            
            if y_i[1] == 0:
                trainY.append([y_i[0],-1.0])
            else:
                trainY.append([y_i[0],1.0])
        print('SVC TRAINING')
        m = Model()
        init_time = time.time()
        m.gridSearch(logspaceC, logspaceGamma, self.lamb, X,trainY, self.validation_split, self.lr,self.epochs, self.culture, verbose=self.verbose_param)
        if self.verbose_param:
                    print(f"--- {time.time() - init_time}s in grid search with C={m.C} and gamma={m.gamma}---")
        return m

    def RFC(self, TS):
        rfc=RandomForestClassifier(random_state=42)
        logspace_max_depth = []
        for i in np.logspace(0,3,self.points):
                logspace_max_depth.append(int(i))
        param_grid = { 
            'n_estimators': [500], #logspace_n_estimators,
            'max_depth' : logspace_max_depth,
            }
        
        CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
        # training set is divided into (X,y)
        TS = np.array(TS, dtype = object)
        X = list(TS[:,0])
        y = list(TS[:,1])
        
        print('RFC TRAINING')
        H = CV_rfc.fit(X,y)

        print(CV_rfc.best_params_)

        return H

    def test(self, model, testSet, out):
        testSet = np.array(testSet, dtype=object)
        XT = list(testSet[:,0])
        yT= list(testSet[:,1])
        yF = []
        yTnew = []
        for i,xT in enumerate(XT):
            yF.append(model.predict(xT, out))
            if yT[i][1] == 0:
                yTnew.append([yT[0],-1.0])
            else:
                yTnew.append([yT[0],1.0])
        cm = confusion_matrix(yT, yF)
        return cm
    
    def save_cm(self, fileName, cm):
        f = FileClass(fileName)
        f.writecm(cm)

    def get_results(self, fileName):
        f = FileClass(fileName)
        return f.readcms()

    def execute(self):
        for i in range(self.times):
            gc.collect()
            print(f'CICLE {i}')
            obj = DS.ds.DSClass()
            obj.mitigation_dataset(self.paths, self.greyscale, 1)
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
            if self.type == 'SVC':
                model = self.SVC(TS)
            elif self.type == 'RFC':
                print('NO RFC IMPLEMENTED UP TO NOW')
                #model = self.RFC(TS)
                break
            else:
                model = self.SVC(TS)
            cms = []
            for k, TestSet in enumerate(TestSets):
                cm = self.test(model, TestSet, k)
                for o in range(3):
                    print(fileNames[k][o])
                    self.save_cm(fileNames[k][o], cm[o])
                    cms.append(cm)
            # Reset Memory each time
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

                  


