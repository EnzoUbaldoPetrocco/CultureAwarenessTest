import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
sys.path.insert(1, '../')
import DS.ds
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

class ClassificatorClass:
    def __init__(self, culture=0, greyscale=0, paths=None,
                 type='SVC', points=50, kernel='linear', times=30):
        self.culture = culture
        self.greyscale = greyscale
        self.paths = paths
        self.type = type
        self.points = points
        self.kernel = kernel
        self.times = times
        

    def prepareDataset(self, paths):
        datasetClass = DS.ds.DSClass()
        datasetClass.build_dataset(paths)
        self.TS = datasetClass.TS
        self.TestSet = datasetClass.TestS

    def SVC(self, TS):
        if self.kernel == 'rbf':
            logspaceC = np.logspace(-2,2,self.points)
            logspaceGamma = np.logspace(-2,2,self.points)
        if self.kernel == 'linear':
            logspaceC = np.logspace(-2,2,self.points)
            logspaceGamma = np.logspace(-2,2,self.points)
        grid = {'C':        logspaceC,
                'kernel':   [self.kernel],
                'gamma':    logspaceGamma}
        MS = GridSearchCV(estimator = SVC(),
                        param_grid = grid,
                        scoring = 'balanced_accuracy',
                        cv = 10,
                        verbose = 0)
        # training set is divided into (X,y)
        TS = np.array(TS, dtype = object)
        X = list(TS[:,0])
        y = list(TS[:,1])
        print('SVC TRAINING')
        H = MS.fit(X,y)
        # Check that C and gamma are not the extreme values
        print(f"C best param {H.best_params_['C']}")
        print(f"gamma best param {H.best_params_['gamma']}")    

        return H

    def RFC(self, TS):
        rfc=RandomForestClassifier(random_state=42)
        logspace_n_estimators = []
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

    def calculate_percentage_confusion_matrix(self, confusion_matrix_list, tot):
                pcms = []
                for i in confusion_matrix_list:
                        true_negative = (i[0,0]/tot)*100
                        false_negative = (i[1,0]/tot)*100
                        true_positive = (i[1,1]/tot)*100
                        false_positive = (i[0,1]/tot)*100
                        pcm = np.array([[true_negative , false_positive],[false_negative, true_positive]])
                        pcms.append(pcm)
                return pcms

    def return_tot_elements(self, cm):
            tot = cm[0,0] + cm[0,1] + cm[1,0] + cm[1,1]
            return tot

    def return_statistics_pcm(self, pcms):
            max_true_negative = 0
            max_false_negative = 0
            max_true_positive = 0
            max_false_positive = 0
            min_true_negative = 100
            min_false_negative = 100
            min_true_positive = 100
            min_false_positive = 100
            count_true_negative = 0
            count_false_negative = 0
            count_true_positive = 0
            count_false_positive = 0
            for i in pcms:
                    true_negative = i[0,0]
                    false_negative = i[1,0]
                    true_positive = i[1,1]
                    false_positive = i[0,1]

                    count_true_negative += true_negative
                    count_false_negative += false_negative
                    count_false_positive += false_positive
                    count_true_positive += true_positive

                    if true_negative > max_true_negative:
                            max_true_negative = true_negative
                    if false_negative > max_false_negative:
                            max_false_negative = false_negative
                    if true_positive > max_true_positive:
                            max_true_positive = true_positive
                    if false_positive > max_false_positive:
                            max_false_positive = false_positive

                    if true_negative < min_true_negative:
                            min_true_negative = true_negative
                    if false_negative < min_false_negative:
                            min_false_negative = false_negative
                    if true_positive < min_true_positive:
                            min_true_positive = true_positive
                    if false_positive < min_false_positive:
                            min_false_positive = false_positive
            
            mean_true_negative = count_true_negative/len(pcms)
            mean_false_negative = count_false_negative/len(pcms)
            mean_true_positive = count_true_positive/len(pcms)
            mean_false_positive = count_false_positive/len(pcms)

            mean_matrix = np.array([[mean_true_negative, mean_false_positive],[mean_false_negative, mean_true_positive]])
            max_matrix = np.array([[max_true_negative, max_false_positive],[max_false_negative, max_true_positive]])
            min_matrix = np.array([[min_true_negative, min_false_positive],[min_false_negative, min_true_positive]])

            matrix = []
            matrix.append(mean_matrix)
            matrix.append(max_matrix)
            matrix.append(min_matrix)
            return matrix
    
    def test(self, model, testSet):
        testSet = np.array(testSet, dtype=object)
        XT = list(testSet[:,0])
        yT= list(testSet[:,1])

        yF = model.predict(XT)
        cm = confusion_matrix(yT, yF)
        return cm
    
    def execute(self):
        results = []
        mixedResults = []
        for i in range(self.times):
            print(f'CICLE {i}')
            obj = DS.ds.DSClass()
            obj.build_dataset(self.paths, self.greyscale)
            # I have to select a culture
            TS = obj.TS[self.culture]
            # I have to test on every culture
            TestSets = obj.TestS
            MixedTestSet = obj.MixedTestS
            if self.type == 'SVC':
                model = self.SVC(TS)
            elif self.type == 'RFC':
                model = self.RFC(TS)
            else:
                model = self.SVC(TS)
            cms = []
            for k, TestSet in enumerate(TestSets):
                cm = self.test(model, TestSet)
                cms.append(cm)
            results.append(cms)
            mixedResults.append(self.test(model, MixedTestSet))
        
        results = np.array(results, dtype = object)
        for i in range(len(obj.TS)):
            result = results[:,i]
            print(f'RESULTS OF CULTURE {i}')
            print(np.shape(result))
            tot = self.return_tot_elements(result[i])
            pcm_list = self.calculate_percentage_confusion_matrix(result, tot)
            statistic = self.return_statistics_pcm(pcm_list)
            for j in statistic:
                print(j)
            accuracy = statistic[0][0][0] + statistic[0][1][1]
            print(f'Accuracy is {accuracy} %')

        print('MIXED RESULTS')
        tot = self.return_tot_elements(mixedResults[0])
        pcm_list = self.calculate_percentage_confusion_matrix(mixedResults, tot)
        statistic = self.return_statistics_pcm(pcm_list)
        for j in statistic:
            print(j)
        accuracy = statistic[0][0][0] + statistic[0][1][1]
        print(f'Accuracy is {accuracy} %')

    def execute_mixed(self):
        results = []
        mixedResults = []
        for i in range(self.times):
            print(f'CICLE {i}')
            obj = DS.ds.DSClass()
            obj.build_dataset(self.paths, self.greyscale)
            # I have to mix the cultures
            TS = []
            for culture in obj.TS:
                  TS.append(culture)
            # I have to test on every culture
            TestSets = obj.TestS
            MixedTestSet = obj.MixedTestS
            if self.type == 'SVC':
                model = self.SVC(TS)
            elif self.type == 'RFC':
                model = self.RFC(TS)
            else:
                model = self.SVC(TS)
            cms = []
            for k, TestSet in enumerate(TestSets):
                cm = self.test(model, TestSet)
                cms.append(cm)
            results.append(cms)
            mixedResults.append(self.test(model, MixedTestSet))
        
        results = np.array(results, dtype = object)
        for i in range(len(obj.TS)):
            result = results[:,i]
            print(f'RESULTS OF CULTURE {i}')
            print(np.shape(result))
            tot = self.return_tot_elements(result[i])
            pcm_list = self.calculate_percentage_confusion_matrix(result, tot)
            statistic = self.return_statistics_pcm(pcm_list)
            for j in statistic:
                print(j)
            accuracy = statistic[0][0][0] + statistic[0][1][1]
            print(f'Accuracy is {accuracy} %')

        print('MIXED RESULTS')
        tot = self.return_tot_elements(mixedResults[0])
        pcm_list = self.calculate_percentage_confusion_matrix(mixedResults, tot)
        statistic = self.return_statistics_pcm(pcm_list)
        for j in statistic:
            print(j)
        accuracy = statistic[0][0][0] + statistic[0][1][1]
        print(f'Accuracy is {accuracy} %')

        


                  


