from cmath import exp
import numpy as np
import math
import random
import time

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
        
    def gradient(self, X, Y, culture):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        otherCultureParts = np.zeros(np.shape(X[0]), dtype=float)
        
        for i in range(np.shape(self.w)[0]):
                otherCultureParts+= self.w[i]
        
        dL = self.first*self.w[culture] + self.second*otherCultureParts + np.matmul(self.third[culture],self.w[culture])-2*np.matmul(X.T,Y[:,1])
        return dL
           
    def fit(self, X, Y, batch_size=10, learning_rate = 0.001, epochs = 1000, grid=False, verbose = False):
        # Adam Algorithm
        # while wt non converged do
        # gt+1 = nabthettaft(thetat-1)
        # mt+1 = beta1*mt + (1-beta1)*(gt)
        # vt+1 = beta2*vt + (1-beta2)*(gt)^2
        # mhat+1 = mt+1/(1-beta1**t)
        # vhat+1 = vt+1/(1-beta2**t)
        # wt+1 = wt - alpha(mhat/(sqrt(vhat)+e))
        # t = t + 1
        q = 3
        n = []
        d = []
        for i in range(q):
            n.append(np.shape(X[i])[0])
            d.append(np.shape(X[i])[1])
        
        mt = []
        vt = []
        self.w = []
        dLdw = []
        for i in range(q):
            zeros = np.zeros(d[0], dtype=float)
            random_numbers = np.random.random(d[0]).astype(float)
            self.w.append(random_numbers)
            mt.append(zeros)
            vt.append(zeros)
            dLdw.append(zeros)

        ids = []
        for c in range(q):
            rnd_sam = np.arange(len(X[c]))
            random.shuffle(rnd_sam)
            ids.append(rnd_sam)
            
        if not grid:
            X = np.asarray(X, dtype=object)
            self.first = 2*self.C + 2*self.lamb
             # Preliminary Computations
            for i in range(q):
                # X^T*X given X n*d matrix
                init_time = time.time()
                for i in range(len(X)):
                    Xcult = np.asarray(X[i], dtype=float)
                    self.third.append(np.matmul(Xcult.T,Xcult))
            if verbose:
                print(f'Time passed to perform matmul: {time.time()-init_time}s')


        self.second =(2*self.lamb*(1-q))/(q*q)
        # Hyperparams
        beta1 = 0.9
        beta2 = 0.9
        e = 0.00000001
        
        for t in range(epochs):
            for i in range(q):                
                dLdw[i] = self.gradient(X[i], Y[i], i)
                
                mt[i] = np.add(beta1*mt[i], (1-beta1)*dLdw[i])
                vt[i] = np.add(beta2*vt[i] , (1-beta2)*(dLdw[i]**2))
                mhat = mt[i]/(1-beta1**(t+1))
                vhatsqr = np.sqrt((vt[i]/(1-beta2**(t+1))).astype(float))
                self.w[i] = self.w[i] - learning_rate*((mhat/vhatsqr)+e) 
            #print(np.sum(dLdw[0]))    
        #self.w = np.asarray(self.w, dtype=object)      
        return self.w
    
    def predict(self, X, out):
        ret = np.sign(np.dot(self.w[out], X))
        return ret
    
    def gridSearch(self, C_list, gamma_list, lamb, X, y, validation_split, learning_rate, epochs, culture_metric=0, classificator = 'linear', verbose = 0):
        # Preliminary Computations
        self.first = 2*self.C + 2*self.lamb
        q = np.shape(X)[0]
        self.culture = culture_metric
        # X^T*X given X n*d matrix
        X = np.asarray(X, dtype=object)
        init_time = time.time()
        self.third = []
        for i in range(len(X)):
            Xcult = np.asarray(X[i], dtype=float)
            self.third.append(2*np.matmul(Xcult.T,Xcult))
        if verbose:
            print(f'Time passed to perform matmul: {time.time()-init_time}s')
        n = np.shape(X[culture_metric])[0]
        ids = []
        for c in range(q):
            rnd_sam = np.arange(len(X[c]), dtype=int)
            random.shuffle(rnd_sam)
            ids.append(rnd_sam)
        max_accuracy = 0
        bestC=-1
        bestGamma=-1
        self.lamb = lamb
        bestW = -1
        for i,C in enumerate(C_list):
            for c,id in enumerate(ids):
                random.shuffle(id)
            self.C = C
            if classificator == 'linear':
                gamma_list = [0]
            for j,gamma in enumerate(gamma_list):
                self.gamma = gamma
                
                train = []
                train_y = []
                
                for c in range(q):
                    trainC = []
                    trainyC = []
                    cultN = int(len(X[c])*(1-validation_split))
                    
                    for l in ids[c][0:cultN]:
                        sample = np.asarray(X[c], dtype=float)
                        sampleY = np.asarray(y[c], dtype=float)
                        trainC.append(sample[l])
                        trainyC.append(sampleY[l])
                    train.append(trainC)
                    train_y.append(trainyC)
                if verbose:
                    print(f'TRAINING C={self.C} and Gamma={self.gamma}')
                init_time = time.time()
                self.fit(train, train_y, learning_rate=learning_rate, epochs=epochs, grid=True, verbose=verbose)
                if verbose:
                    print(f"--- {time.time() - init_time}s in fitting with C={C} and gamma={gamma}---")
                count = 0
                val_numb = int(len(ids[culture_metric])*(validation_split))
                
                init_time = time.time()
                for l in ids[culture_metric][cultN:val_numb+cultN]:
                    Xcult = np.asarray(X[culture_metric], dtype=float)[l]
                    yP = self.predict(Xcult, culture_metric)
                    y_i = np.asarray(y[culture_metric], dtype=float)[l][1]
                    if yP == y_i:
                        count = count + 1
                if verbose:
                    print(f"--- {time.time() - init_time}s in predicting {val_numb} samples---")
                accuracy = count/val_numb
                if verbose:
                    print(f"Accuracy is {accuracy*100}%")
                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    bestC = C
                    bestGamma = gamma
                    bestW = self.w
        if verbose:
            print(f'max validation accuracy is: {max_accuracy}')
        self.C = bestC
        self.gamma = bestGamma
        
        self.w = bestW

