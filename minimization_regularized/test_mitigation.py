import sys
sys.path.insert(1, '../')
import numpy as np
from standard_from_scratch.regularized import Model
from sklearn.metrics import confusion_matrix

# Let's say we have a two random sets. I want to test if
# the models can predict this sets.
# Let's notice that i want 3 cultures so 6 sets.
n_samples = 100
var_param = 100
# First class belonging to the first culture
a0 = np.random.multivariate_normal([10, 10, 10], [[var_param, 0, 0],[0, var_param, 0], [0, 0, var_param]], n_samples)
# Second class belonging to the first culture
a1 = np.random.multivariate_normal([10, -10, 10], [[var_param, 0, 0],[0, var_param, 0], [0, 0, var_param]], n_samples)
# First class belonging to the second culture
b0 = np.random.multivariate_normal([-10, -10, 10], [[var_param, 0, 0],[0, var_param, 0], [0, 0, var_param]], n_samples)
# Second class belonging to the second culture
b1 = np.random.multivariate_normal([-10, 10, 10], [[var_param, 0, 0],[0, var_param, 0], [0, 0, var_param]], n_samples)
# First class belonging to the third culture
c0 = np.random.multivariate_normal([-10, -10, -10], [[var_param, 0, 0],[0, var_param, 0], [0, 0, var_param]], n_samples)
# Second class belonging to the third culture
c1 = np.random.multivariate_normal([-10, 10, -10], [[var_param, 0, 0],[0, var_param, 0], [0, 0, var_param]], n_samples)



def test(XT, yT, m, out):
    yP = []
    for i,xT in enumerate(XT):
        yPi = m.predict(xT, out)
        yP.append(yPi)
    cm = confusion_matrix(yT, yP)
    tot = 0
    for i in cm:
        for j in i:
            tot += j
    cm = cm / tot
    #print(cm)
    print(f'Accuracy is {cm[0][0] + cm[1][1]}')
    return cm
    

points = 10
learning_rate = 0.01
epochs = 1000
validation_split = 0.2
test_split = 0.2
n = int(len(a0*6)*(1-test_split))
nT = int(len(a0*6)*(test_split))
Clist = np.logspace(-2,2,points)
Gammalist = np.logspace(-2,2,points)
Lamblist = np.logspace(-2,2,points)
underrepr_param = 0.1

# TESTING CULTURE
ya0 = [[0, -1]] * n
ya1 = [[0, 1]] * n
yb0 = [[1,-1]] * int(n*underrepr_param)
yb1 = [[1,1]] * int(n*underrepr_param)
yc0 = [[2,-1]] * int(n*underrepr_param)
yc1 = [[2,1]] * int(n*underrepr_param)

ya0T = [-1] * nT
ya1T = [1] * nT
yb0T = [-1] * nT
yb1T = [1] * nT
yc0T = [-1] * nT
yc1T = [1] * nT
X = []
y = []

XT0 = [*a0[n:n + nT], *a1[n:n + nT]]
XT1 = [*b0[n:n + nT], *b1[n:n + nT]]
XT2 =  [*c0[n:n + nT], *c1[n:n + nT]]
train = [*a0[0:n],  *a1[0:n], *b0[0:int(n*underrepr_param)], *b1[0:int(n*underrepr_param)], *c0[0:int(n*underrepr_param)], *c1[0:int(n*underrepr_param)]]
train_y = [*ya0, *ya1, *yb0, *yb1, *yc0, *yc1]
temp = list(zip(train, train_y))
np.random.shuffle(temp)
train, train_y = zip(*temp)
for i in range(len(train)):
    X.append(train[i].flatten())
    y.append(train_y[i])

m = Model()
m.gridSearch(Clist, Gammalist, 0, X, y, validation_split, learning_rate, epochs, culture_metric=0)
# TEST
print('LAMBDA = 0')
print('Out 0 on Culture 0')
test(XT0, ya0T + ya1T, m, 0)
print('Out 0 on Culture 1')
test(XT0, yb0T + yb1T, m, 0)
print('Out 0 on Culture 2')
test(XT0, yc0T + yc1T, m, 0)

print('Out 1 on Culture 0')
test(XT0, ya0T + ya1T, m, 1)
print('Out 1 on Culture 1')
test(XT0, yb0T + yb1T, m, 1)
print('Out 1 on Culture 2')
test(XT0, yc0T + yc1T, m, 1)

print('Out 2 on Culture 0')
test(XT0, ya0T + ya1T, m, 2)
print('Out 2 on Culture 1')
test(XT0, yb0T + yb1T, m, 2)
print('Out 2 on Culture 2')
test(XT0, yc0T + yc1T, m, 2)

# TESTING RESULTS IN MIXED WITH LOW LAMBDA
m = Model()
m.gridSearch(Clist, Gammalist, 0.01, X, y, validation_split, learning_rate, epochs, culture_metric=0)
# TEST
print('LAMBDA = 0.01')
print('Out 0 on Culture 0')
test(XT0, ya0T + ya1T, m, 0)
print('Out 0 on Culture 1')
test(XT0, yb0T + yb1T, m, 0)
print('Out 0 on Culture 2')
test(XT0, yc0T + yc1T, m, 0)

print('Out 1 on Culture 0')
test(XT0, ya0T + ya1T, m, 1)
print('Out 1 on Culture 1')
test(XT0, yb0T + yb1T, m, 1)
print('Out 1 on Culture 2')
test(XT0, yc0T + yc1T, m, 1)

print('Out 2 on Culture 0')
test(XT0, ya0T + ya1T, m, 2)
print('Out 2 on Culture 1')
test(XT0, yb0T + yb1T, m, 2)
print('Out 2 on Culture 2')
test(XT0, yc0T + yc1T, m, 2)

# TESTING RESULTS IN MIXED WITH HIGH LAMBDA
m = Model()
m.gridSearch(Clist, Gammalist, 100, X, y, validation_split, learning_rate, epochs, culture_metric=0)
# TEST
print('LAMBDA = 100')
print('Out 0 on Culture 0 confusion matrix is')
test(XT0, ya0T + ya1T, m, 0)
print('Out 0 on Culture 1 confusion matrix is')
test(XT0, yb0T + yb1T, m, 0)
print('Out 0 on Culture 2 confusion matrix is')
test(XT0, yc0T + yc1T, m, 0)

print('Out 1 on Culture 0 confusion matrix is')
test(XT0, ya0T + ya1T, m, 1)
print('Out 1 on Culture 1 confusion matrix is')
test(XT0, yb0T + yb1T, m, 1)
print('Out 1 on Culture 2 confusion matrix is')
test(XT0, yc0T + yc1T, m, 1)

print('Out 2 on Culture 0 confusion matrix is')
test(XT0, ya0T + ya1T, m, 2)
print('Out 2 on Culture 1 confusion matrix is')
test(XT0, yb0T + yb1T, m, 2)
print('Out 2 on Culture 2 confusion matrix is')
test(XT0, yc0T + yc1T, m, 2)
