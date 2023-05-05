from keras.datasets import mnist
import sys
sys.path.insert(1, '../')
from standard_from_scratch.svm import SVM
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
import numpy as np

(train_X, train_y), (test_X, test_y) = mnist.load_data()
train_X = train_X[0:300]
train_y = train_y[0:300]
test_X = test_X[0:30]
test_y = test_y[0:30]
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

yF_sk_list.append(m_sk.predict(XT)[0])
#print(np.shape(test_X))
#print(np.shape(y_i))
#print(np.shape(yF_list))
#print(y_i)
#print(yF_list)
cm = confusion_matrix(test_y, yF_list)
print(cm)
# compare with model from sklearn
cm_sk = confusion_matrix(test_y, yF_sk_list)
print(cm_sk)