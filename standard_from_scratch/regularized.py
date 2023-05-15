from cmath import exp
import numpy as np
import math
import random

## Formula is in paper

class Model:
    def __init__(self, C=1.0, kernel = 'linear', gamma = 1.0, lamb = 0):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.lamb = lamb

    def K(self, x, x_i):
        if self.kernel == 'linear':
            return sum(x+x_i)
        elif self.kernel == 'exp':
            return exp(-self.gamma * sum((x - x_i)^2))
        elif self.kernel == 'rbf':
            return exp(-self.gamma*np.linalg.norm(x,x_i)**2)
    
    def l1(self, w, b, x, y):
        y_predicted = []
        for i in range(x.shape[0]):
            y_predicted.append((np.dot(w,x[i]) + b)[0])
        loss = np.sum(np.abs(y-y_predicted))
        return loss
    
    def projection(self, x):
        if self.kernel == 'linear':
            return x
        if self.kernel == 'rbf':
            return math.exp(self.gamma*(x**2))
        
    def gradient(self, q, X, Y, culture):
        otherCultureParts = 0
        for i in range(np.shape(self.w)[0]):
            if culture != i:
                otherCultureParts+= (-2*self.lamb/q)*self.w[i]
        dL = (((2*q*self.lamb-2*self.lamb+q)/q) + 2*self.C*np.dot(X.T,X))*self.w[culture] - 2*self.C*Y*X.T + otherCultureParts
        #print(dL)
        return dL
        
    
    def fit(self, X, Y, batch_size=10, learning_rate = 0.001, epochs = 2):
        # Adam Algorithm
        # while wt non converged do
        # mt+1 = beta1*mt + (1-beta1)*(dL/dwit)
        # vt+1 = beta2*vt + (1-beta2)*(dL/dwit)
        # mhat+1 = mt+1/(1-beta1**t)
        # vhat+1 = vt+1/(1-beta2**t)
        # wt+1 = wt - alpha((mhat+1/sqrt(vhat+1))+e)
        # t = t + 1
        q = 3
        n = np.shape(X)[0]
        d = np.shape(X)[1]
        mt = []
        vt = []
        mhat = []
        vhat = []
        self.w = []
        dLdw = []
        for i in range(q):
            zeros = np.zeros(d)
            random_numbers = np.random.random(d)
            self.w.append(random_numbers)
            mt.append(zeros)
            vt.append(zeros)
            mhat.append(zeros)
            vhat.append(zeros)
            dLdw.append(zeros)

        ids = np.arange(n)
        random.shuffle(ids)

        
        # Hyperparams
        beta1 = 0.8
        beta2 = 0.9
        e = 0.00000001
        
        for t in range(epochs):
            for j in range(n):
                x = ids[j]
                i = Y[x][1] # referring culture
                dLdw[i] = self.gradient(q, X[x], Y[x][0], i)
                mt[i] = beta1*mt[i] + (1-beta1)*dLdw[i]
                vt[i] = beta2*vt[i] + (1-beta2)*(dLdw[i]**2)
                mhat[i] = mt[i]/(1-beta1**(t+1))
                vhat[i] = vt[i]/(1-beta2**(t+1))
                self.w[i] = self.w[i] - learning_rate*((mhat[i]/np.sqrt(vhat[i]))+e)     
        #self.w = np.asarray(self.w, dtype=object)      
        return self.w
    
    def predict(self, X, out):
        ret = np.sign(np.dot(self.w[out], X))
        return ret
    
    def gridSearch(self, C_list, gamma_list, lamb, X, y, validation_split, learning_rate, epochs, culture_metric=0):
        n = np.shape(X)[0]
        ids = np.arange(n)
        max_accuracy = 0
        bestC=-1
        bestGamma=-1
        self.lamb = lamb
        for i,C in enumerate(C_list):
            random.shuffle(ids)
            self.C = C
            for j,gamma in enumerate(gamma_list):
                self.gamma = gamma
                
                train = []
                train_y = []
                for l in ids[0:int(len(ids)*(1-validation_split))]:
                    train.append(X[ids[l]])
                    train_y.append(y[ids[l]])
                self.fit(train, train_y, learning_rate=learning_rate, epochs=epochs)
                count = 0
                val_numb = int(len(ids)*(validation_split))
                #print(val_numb)
                for l in ids[0:val_numb]:
                    yP = self.predict(X[ids[l]], culture_metric)
                    if yP == y[ids[l]][culture_metric]:
                        count+=1
                accuracy = count/val_numb
                
                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    bestC = C
                    bestGamma = gamma

        self.C = bestC
        self.gamma = bestGamma
        #print('Weights per culture:')
        #for i in self.w:
        #    print(i)

