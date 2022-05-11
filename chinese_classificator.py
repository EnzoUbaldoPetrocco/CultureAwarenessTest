#! /usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import manipulating_images_better
from math import floor
from sklearn.metrics import confusion_matrix
############################################################
############### READ DATA ##################################

itd = manipulating_images_better.ImagesToData()
itd.bf_ml()

CX = itd.chinese[0:floor(len(itd.chinese)*0.7)]
CXT = itd.chinese[floor(len(itd.chinese)*0.7):len(itd.chinese)-1]
CY = itd.chinese_categories[0:floor(len(itd.chinese)*0.7)]
CYT = itd.chinese_categories[floor(len(itd.chinese)*0.7):len(itd.chinese)-1]

FX = itd.french[0:floor(len(itd.french)*0.7)]
FXT = itd.french[floor(len(itd.french)*0.7):len(itd.french)-1]
FY = itd.french_categories[0:floor(len(itd.french)*0.7)]
FYT = itd.french_categories[floor(len(itd.french)*0.7):len(itd.french)-1]


MX = itd.mixed[0:floor(len(itd.mixed)*0.7)]
MXT = itd.mixed[floor(len(itd.mixed)*0.7):len(itd.mixed)-1]
MY = itd.mixed_categories[0:floor(len(itd.mixed)*0.7)]
MYT = itd.mixed_categories[floor(len(itd.mixed)*0.7):len(itd.mixed)-1]

####################################################################
###################### PLOT IMAGE ##################################
print('images')
plt.figure()
plt.imshow(np.reshape(CX[30], (itd.size,itd.size)))
plt.show()
####################################################################
################### NORMALIZE DATA #################################
print('NORMALIZE DATA')
scalerX = preprocessing.MinMaxScaler()
CX = scalerX.fit_transform(CX)
CXT = scalerX.transform(CXT)

#####################################################################
################### MODEL SELECTION (HYPERPARAMETER TUNING)##########
print('MODEL SELECTION AND TUNING')
Cgrid = {'C':        np.logspace(-4,3,5),
        'kernel':   ['rbf'],
        'gamma':    np.logspace(-4,3,5)}
CMS = GridSearchCV(estimator = SVC(),
                  param_grid = Cgrid,
                  scoring = 'balanced_accuracy',
                  cv = 10,
                  verbose = 0)
CH = CMS.fit(CX,CY)

print('CLASSIFICATION')
CM = SVC(C = CH.best_params_['C'],
        kernel = CH.best_params_['kernel'],
        gamma = CH.best_params_['gamma'])
CM.fit(CX,CY)

####################################################
################## TESTING #########################
print('Predicting Chinese test set')
CYF = CM.predict(CXT)
print(confusion_matrix(CYT,CYF))

print('Predicting French test set')
CFYF = CM.predict(FXT)
print(confusion_matrix(FYT,CFYF))

print('PREDICTING MIX TEST SET')
MCYF = CM.predict(MXT)
print(confusion_matrix(MYT,MCYF))


print('arrivato')