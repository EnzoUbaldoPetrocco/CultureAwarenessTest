from cmath import exp
import numpy as np
import pandas

class SVM:
    def __init__(self, C=1.0, kernel = 'linear', gamma = 1.0):
        self.C = C
        self.w = 0
        self.b = 0
        self.kernel = kernel
        self.gamma = gamma

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
        
    
    def hingeloss(self, w, b, x, y):
        predicted = []
        for i in range(np.shape(x)[0]):
            predicted.append((np.dot(w,x[i]) + b))
        hinge_loss = np.mean([max(0, 1-x*y) for x, y in zip(y, predicted)])
        return hinge_loss
    
    def fit(self, X, Y, batch_size=10, learning_rate = 0.01, epochs = 1000):
        number_of_features = X.shape[1]
        number_of_samples = X.shape[0]
        c = self.C
        ids = np.arange(number_of_samples)
        np.random.shuffle(ids)
        
        for i, y in enumerate(Y):
            if y <= 0:
                Y[i] = -1
        w = np.zeros(number_of_features)
        b = 0
        losses = []
        count = 0
        for i in range(epochs):
            #if self.kernel == 'linear':
            #    l = self.hingeloss(w,b,X,Y)
            #else:
            #    l = self.l1(w,b,X,Y)
            l = self.hingeloss(w,b,X,Y)
            losses.append(l)
            for batch_initial in range(0, number_of_samples, batch_size):
                gradw = 0
                gradb = 0
                for j in range(batch_initial, batch_initial + batch_size):
                    if j < number_of_samples:
                        x = ids[j]

                        ti = Y[x]*(np.dot(w, X[x]) + b)
                        #print(np.shape(ti))
                        if ti>=1:
                            gradw+=0
                            gradb+=0
                            count+=1
                        else:
                            gradw+= w - c* Y[x]*X[x]
                            gradb+=c*Y[x]
                w = w - learning_rate * gradw
                b = b - learning_rate * gradb
                learning_rate = learning_rate * 0.1
        self.w = w
        self.b = b
        #print(losses)
        print(f'Count is {count}')
        return self.w, self.b, losses
    
    def predict(self, X):
        ret = np.sign(np.dot(self.w.T, X) + self.b)
        return ret
    

