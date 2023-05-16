from keras.datasets import mnist
import sys
sys.path.insert(1, '../')
from standard_from_scratch.svm import SVM
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
import numpy as np

(train_X, train_y), (test_X, test_y) = mnist.load_data()
train_X = [x for i,x in enumerate(train_X) if train_y[i] == 0 or train_y[i] == 1 ]
train_y = [y for i,y in enumerate(train_y) if train_y[i] == 0 or train_y[i] == 1 ]
test_X = [x for i,x in enumerate(test_X) if test_y[i] == 0 or test_y[i] == 1 ]
test_y = [y for i,y in enumerate(test_y) if test_y[i] == 0 or test_y[i] == 1 ]

i1 = 800
i2 = 200
train_X = train_X[0:i1]
train_y = train_y[0:i1]
test_X = test_X[i1:i1+i2]
test_y = test_y[i1:i1+i2]
for i,y in enumerate(test_y):
    if y==0:
        test_y[i]=-1
#print('X_train: ' + str(train_X.shape))
#print('Y_train: ' + str(train_y.shape))
#print('X_test:  '  + str(test_X.shape))
#print('Y_test:  '  + str(test_y.shape))
m = SVM()
X = []
for sample in train_X:
    X.append(sample.flatten())
X = np.array(X, dtype=object)
m.fit(X, train_y)
yF_list = []
yF_sk_list = []
m_sk = SVC(C=m.C, kernel='linear')
m_sk.fit(X, train_y)
XT = []

for sample in test_X:
    y_i = m.predict(sample.flatten())
    yF_list.append(y_i)
    XT.append(sample.flatten())

yF_sk_list= m_sk.predict(XT)
for i, y in enumerate(yF_sk_list):
    if y==0:
        yF_sk_list[i] = -1

cm = confusion_matrix(test_y, yF_list)
print(cm)
# compare with model from sklearn
cm_sk = confusion_matrix(test_y, yF_sk_list)
print(cm_sk)

