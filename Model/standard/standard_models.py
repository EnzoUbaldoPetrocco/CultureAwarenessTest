#!/usr/bin/env python
__author__ = "Enzo Ubaldo Petrocco"
import gc
import sys

from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

sys.path.insert(1, "../")
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from Model.GeneralModel import GeneralModelClass
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import v2
import torchvision.transforms.functional as TF


class StandardModels(GeneralModelClass):
    def __init__(
        self,
        type="SVC",
        points=50,
        kernel="linear",
        verbose_param=0,
        learning_rate=1e-3,
        epochs=15,
        batch_size=1,
    ):
        """
        Initialization function for modeling standard ML models.
        We have narrowed the problems to image classification problems.
        I have implemented SVM (with linear and gaussian kernel) and Random Forest, using scikit-learn library;
        ResNet using Tensorflow library.
        :param type: selects the algorithm "SVC", "RFC" and "RESNET" are possible values.
        :param points: n of points in gridsearch for SVC and RFC
        :param kernel: type of kernel for SVC: "linear" and "gaussian" are possible values.
        :param verbose_param: if enabled, the program logs more information
        :param learning_rate: hyperparameter for DL
        :param epochs: hyperparameter for DL
        :param batch_size: hyperparameter for DL
        """
        GeneralModelClass.__init__(self)
        self.type = type
        self.points = points
        self.kernel = kernel
        self.verbose_param = verbose_param
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

    def SVC(self, TS):
        """
        This function performs the model selection on SVM for Classification
        :param TS: union between training and validation set
        :return the best model
        """
        if self.kernel == "rbf":
            logspaceC = np.logspace(-4, 3, self.points)  # np.logspace(-2,2,self.points)
            logspaceGamma = np.logspace(
                -4, 3, self.points
            )  # np.logspace(-2,2,self.points)
            grid = {"C": logspaceC, "kernel": [self.kernel], "gamma": logspaceGamma}
        if self.kernel == "linear":
            logspaceC = np.logspace(-4, 3, self.points)  # np.logspace(-2,2,self.points)
            logspaceGamma = np.logspace(
                -4, 3, self.points
            )  # np.logspace(-2,2,self.points)
            grid = {"C": logspaceC, "kernel": [self.kernel]}

        MS = GridSearchCV(
            estimator=SVC(),
            param_grid=grid,
            scoring="balanced_accuracy",
            cv=10,
            verbose=self.verbose_param,
        )
        # training set is divided into (X,y)
        TS = np.array(TS, dtype=object)
        del TS
        X = list(TS[:, 0])
        y = list(TS[:, 1])
        print("SVC TRAINING")
        H = MS.fit(X, y)
        # Check that C and gamma are not the extreme values
        print(f"C best param {H.best_params_['C']}")
        # print(f"gamma best param {H.best_params_['gamma']}")
        self.model = H

    def RFC(self, TS):
        """
        This function performs the model selection on Random Forest for Classification
        :param TS: union between training and validation set
        :return the best model
        """
        rfc = RandomForestClassifier(random_state=42)
        logspace_max_depth = []
        for i in np.logspace(0, 3, self.points):
            logspace_max_depth.append(int(i))
        param_grid = {
            "n_estimators": [500],  # logspace_n_estimators,
            "max_depth": logspace_max_depth,
        }

        CV_rfc = GridSearchCV(
            estimator=rfc, param_grid=param_grid, cv=5, verbose=self.verbose_param
        )
        # training set is divided into (X,y)
        TS = np.array(TS, dtype=object)
        X = list(TS[:, 0])
        y = list(TS[:, 1])
        del TS
        print("RFC TRAINING")
        H = CV_rfc.fit(X, y)
        # print(CV_rfc.best_params_)
        self.model = H

    def newModelSelection(self, TS, VS, aug, show_imgs=False, batches=[32], lrs=[1e-3, 1e-4, 1e-5], fine_lrs=[1e-6, 1e-7], epochs=25, fine_epochs=5, nDropouts=[0.4], g=0.1):
        best_loss = np.inf
        for b in batches:
            for lr in lrs:
                for fine_lr in fine_lrs:
                    for nDropout in nDropouts:
                            self.model= None
                            torch.cuda.empty_cache()
                            gc.collect()
                            print(f"Training with: batch_size={b}, lr={lr}, fine_lr={fine_lr}, nDropout={nDropout}")
                            loss = self.newDL(TS, VS, aug, show_imgs, b, lr, fine_lr, epochs, fine_epochs, nDropout)
                            if loss < best_loss:
                                best_loss = loss
                                best_bs = b
                                best_lr = lr
                                best_fine_lr = fine_lr
                                best_nDropout = nDropout
            print(f"Best loss:{best_loss}, best batch size:{best_bs}, best lr:{best_lr}, best fine_lr:{best_fine_lr}, best_dropout:{best_nDropout}")
            TS = TS + VS
            self.newDL(TS, None, aug, show_imgs, best_bs, best_lr, best_fine_lr, epochs, fine_epochs, best_nDropout, val=False)

    def train_model(self, model, criterion, optimizer, dataloaders, device, n_images, num_epochs=3, aug=False, transforms=None, val=False):
        if val:
            phases = ['train', 'validation']
        else:
            phases = ['train']

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch+1, num_epochs))
            print('-' * 10)

            for phase in ['train', 'validation']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.type(torch.LongTensor).to(device)
                    
                    if aug and phase=='train':
                        inputs = transforms(inputs)
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    _, preds = torch.max(outputs, 1)
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                
                epoch_loss = running_loss / n_images[phase]
                epoch_acc = running_corrects.double() / n_images[phase]

                if phase=='validation':
                    val_loss = epoch_loss

                print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                            epoch_loss,
                                                            epoch_acc))
                
        self.model = model
        model = None
        
        return val_loss


    def newDL(self, TS, VS, aug=False, show_imgs=False, batch_size=32, lr = 1e-3, fine_lr = 1e-5, epochs=5, fine_epochs=5, nDropout = 0.2, g=0.1, val=True):
        if val:
            n_images = {'train':len(TS[0]), 'validation':len(VS[0])}
        else:
            n_images = {'train':len(TS[0])}
        shape = np.shape(TS[0][0])

        X = torch.Tensor(TS[0]) # transform to torch tensor
        y = torch.Tensor(TS[1])
        TS = TensorDataset(X,y) # create your datset
        if val:
            Xv = torch.Tensor(VS[0]) # transform to torch tensor
            yv = torch.Tensor(VS[1])
            VS = TensorDataset(Xv,yv) # create your datset
        

            dataloaders = {
                'train':
                DataLoader(TS,
                    batch_size=32,
                    shuffle=True,),  # for Kaggle
                'validation':
                DataLoader(VS,
                    batch_size=32,
                    shuffle=False,)  # for Kaggle
            }
        else:
            dataloaders = {
                'train':
                DataLoader(TS,
                    batch_size=32,
                    shuffle=True,),  # for Kaggle
            }


        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        for inputs, labels in dataloaders['train']:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    #print(f"Shape of dataloaders['train'][0]={np.shape(inputs)}")
                    #print(f"Shape of dataloaders['train'][1]={np.shape(labels)}")

        
        transforms = v2.Compose([
            #v2.RandomResizedCrop(size=(int(shape[0]*(1-g)), int(shape[1]*(1-g))), antialias=True),
            v2.RandomCrop(int(shape[1]*(1-g))),
            v2.RandomHorizontalFlip(p=g),
            v2.RandomPerspective(distortion_scale=g, p=g),
            v2.RandomPhotometricDistort(p=g),
            v2.ToDtype(torch.float32, scale=True),
        ])
        samples = []
        for i in range(9):
            samples.append( X[np.random.randint(0, len(X)-1)])
        if show_imgs:
            plt.figure(figsize=(10,10))
            for i, image in enumerate(samples):
                ax = plt.subplot(3, 3, i+1)
                if aug:
                    image = transforms(image)
                image = torch.moveaxis(image, 0, -1)

                plt.imshow(image)
                plt.axis("off")
            plt.show()

        model = models.resnet50(pretrained=True).to(device)
    
        for param in model.parameters():
            param.requires_grad = False   
            
        model.fc = nn.Sequential(
                    nn.Linear(2048, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, 2),
                    
                    ).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.fc.parameters())

        #for param in model.parameters():
        #    param.requires_grad = False

        ct = 0
        for child in model.children():
            ct += 1
            if ct < 10:
                for param in child.parameters():
                    param.requires_grad = False

        #verify
        #for name, param in model.named_parameters():
        #    print(name,param.requires_grad)

        val_loss = self.train_model(model, criterion, optimizer, dataloaders, device, n_images, num_epochs=epochs, aug=aug, transforms=transforms, val=val)

        for param in model.parameters():
            param.requires_grad = True


        val_loss = self.train_model(model, criterion, optimizer, dataloaders, device, n_images, num_epochs=fine_epochs, aug=aug, transforms=transforms, val=val)
        self.device = device

        return  val_loss



    def fit(
        self, TS, VS=None, adversary=0, eps=0.05, mult=0.2, gradcam=False, out_dir="./", complete = 0, aug=0, g=0.1
    ):
        """
        General function for implementing model selection
        :param TS: training set
        :param VS: validation set
        :param adversary: if enabled, adversarial training is enabled
        :param eps: if adversary enabled, step size of adversarial training
        :param mult: if adversary enabled, multiplier of adversarial training
        :param gradcam: if enabled, gradcam callback is called
        :param out_dir: if gradcam enabled, output directory of gradcam heatmap
        :param complete: dummy argument
        """
        if self.type == "SVC":
            self.SVC(TS)
        elif self.type == "RFC":
            self.RFC(TS)
        elif self.type == "DL" or "RESNET":
            self.newModelSelection(TS, VS, aug=aug, g=g)
            """self.DL_model_selection(
                TS, VS, adversary, eps, mult, gradcam=gradcam, out_dir=out_dir
            )"""
        else:
            self.newModelSelection(TS, VS, aug=aug, g=g)
