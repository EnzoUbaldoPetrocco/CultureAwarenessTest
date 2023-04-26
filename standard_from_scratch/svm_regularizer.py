import numpy as np
class SVM:
    def __init__(self, C=1.0):
        self.C = C
        self.w = 0
        self.b = 0
    
    def hingeloss(self, w, b, x, y):
        # 1/2*||w||^2
        reg = 0.5 * (w * w)
        loss = 0
        for i in range(x.shape[0]):
            # y_i*((w . x_i) + b)
            opt_term = y[i] * ((np.dot(w, x[i])) + b)
            loss = loss + reg + self.C * max(0, 1-opt_term)
        return loss
    
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
            l = self.hingeloss(w,b,X,Y)
            losses.append(l)
            for batch_initial in range(0, number_of_samples, batch_size):
                gradw = 0
                gradb = 0
                for j in range(batch_initial, batch_initial + batch_size):
                    if j < number_of_samples:
                        x = ids[j]
                        ti = Y[x]*(np.dot(w, X[X].T) + b)
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
        return np.sign(prediction)
    

