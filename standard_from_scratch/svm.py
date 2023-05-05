from cmath import exp
import numpy as np
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
            y_predicted.append(y[i]*(np.dot(w,x[i]) + b)[0])
        loss = np.sum(np.abs(y-y_predicted))
        return loss
        
    
    def hingeloss(self, w, b, x, y):
        y_predicted = []
        for i in range(x.shape[0]):
            y_predicted.append(y[i]*(np.dot(w,x[i]) + b))
        
        hinge_loss = np.mean([max(0, 1-x*y_i) for x, y_i in zip(y_predicted, y)])
        return hinge_loss
    
    def fit(self, X, Y, batch_size=10, learning_rate = 0.001, epochs = 1000):
        number_of_features = X.shape[1]
        number_of_samples = X.shape[0]
        c = self.C
        ids = np.arange(number_of_samples)
        np.random.shuffle(ids)

        w = np.zeros((1, number_of_features))
        b = 0
        losses = []
        for i in range(epochs):
            if self.kernel == 'linear':
                l = self.hingeloss(w,b,X,Y)
            else:
                l = self.l1(w,b,X,Y)
            losses.append(l)
            for batch_initial in range(0, number_of_samples, batch_size):
                gradw = 0
                gradb = 0
                for j in range(batch_initial, batch_initial + batch_size):
                    if j < number_of_samples:
                        x = ids[j]
                        #print(np.shape(b))
                        #print(np.shape(w))
                        #print(np.shape(X[X]))
                        #print(np.shape((np.dot(w[0], X[x][0]) + b)))
                        #print(np.shape(Y[x]))
                        ti = Y[x]*(np.dot(w, X[x]) + b)
                        #print(np.shape(ti))
                        if ti>1:
                            gradw+=0
                            gradb+=0
                        else:
                            gradw+= c* Y[x]*X[x]
                            gradb+=c*Y[x]
                
                w = w - learning_rate* w + learning_rate * gradw
                b = b + learning_rate * gradb
        
        self.w = w
        self.b = b

        return self.w, self.b, losses
    
    def predict(self, X):
        prediction = np.dot(X,self.w[0]) + self.b
        #print(f'shape of X: {np.shape(X)}')
        #print(f'Shape of self.w[0]: {np.shape(self.w[0])}')
        #print(f'Shape of self.b: {np.shape(self.b)}')
        #print(f'Shape of prediction {np.shape(prediction)}')
        return np.sign(prediction)
    

