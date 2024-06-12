#!/usr/bin/env python
__author__ = "Enzo Ubaldo Petrocco"
from sklearn.metrics import confusion_matrix
import numpy as np
from matplotlib import pyplot as plt
import cv2
import torch
import gc


class GeneralModelClass:
    """
    This Class is the middleware for collecting common actions of the models
    """
    def __init__(self) -> None:
        """
        Init function links self.model attribute
        """
        self.model = 0

    def __call__(self, X, out=-1):
        """
        Call function makes the inference according to 
        the model (standard, our Mitigation Strategy)
        :param X: samples from which we make the inference 
        :param out: if not -1, we select the output 
        :return list of inferences
        """
        if self.model != None:
            if out>0:
                yP = self.model(X)[:,out]
                _, yP = torch.max(yP, 1)
                print('Mannagghia')
                print(f"Shape of yP is {np.shape(yP)}")
                return yP
            else:
                yP = self.model(X)
                _, yP = torch.max(yP, 1)
                return yP
        else:
            print("Try fitting the model before")
            return None

    def quantize(self, yF):
        """
        Quantize a prediction, because we are dealing with binary classification.
        In principle we could set a threshold for imbalanced learning. 
        Since the imbalance is not inter class, but intra class, we simply set the threshold to (max-min)/2=0.5
        :return the prediction quantized
        """
        values = []
        for y in yF:
            if y > 0.5:
                values.append(1)
            else:
                values.append(0)
        return values

    def test(self, Xt, out=-1):
        """
        Test the quality of the model on a set of samples
        :param Xt: set of samples
        :param out: desired output to be tested if any  

        :return list of quantized predictions of the model
        """
        
        if self.model:
            yF = self(Xt, out)
            yFq = self.quantize(yF)
            return yFq
        else:
            print("Try fitting the model before")
            return None

    def get_model_stats(self, Xt, yT, out=-1):
        """
        This function returns a confusion matrix based on a set of samples and its true values
        :param Xt: set of samples
        :param yT: true values of Xt
        :param out: desired output to be tested if any

        :return list confusion matrix
        """
        gc.collect()
        Xt = torch.Tensor(Xt).to(self.device)
        torch.cuda.empty_cache()
        gc.collect()
        yFq = self.test(Xt, out)
        if len(np.shape(yT)) > 1:
            if type(yT) == np.ndarray:
                yT = yT[:, 1]
            elif type(yT) == list:
                yT = np.asarray(yT)[:, 1]
        
        if yFq:
            # yT = list([c_i, y_i])
            cm = confusion_matrix(y_true=yT, y_pred=yFq)
            return cm
        