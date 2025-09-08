#!/usr/bin/env python
__author__ = "Enzo Ubaldo Petrocco"
import sys

sys.path.insert(1, "../")
import tensorflow as tf
tf_version = tf.__version__
# Split version string into major, minor, and patch numbers
version_tuple = tuple(map(int, tf_version.split(".")))
from Model.diffusion.diffusion_standard import DiffusionStandardModel
# Check if the version is less than 2.15
"""if version_tuple[1] < 15:
    from Model.diffusion.diffusion_standard import DiffusionStandardModel
else:
    from Model.diffusion.diffusion_standard_new_tf import DiffusionStandardModel"""
from Model.mitigated.mitigated_models import MitigatedModels
from Model.mitigated.mitigated_models_advanced import MitigatedModels as MitigatedModelsAdvanced
from Model.standard.standard_models import StandardModels
from Model.standard.gradcam_standard import StandardModels4GradCam
from Model.adversarial.adversarial import AdversarialStandard
from Model.discriminator.discriminator import Discriminator
from Utils.Data.Data import DataClass
from Utils.FileManager.FileManager import FileManagerClass
from Utils.Results.Results import ResultsClass
from Utils.Data.deep_paths import DeepStrings
from Utils.Data.shallow_paths import ShallowStrings
from Utils.Data.Data import PreprocessingClass
import numpy as np

import os
import gc
import cv2
import time
import random

class NullWriter:
    def write(self, _): pass

def suppress_output():
    sys.stdout = NullWriter()

def restore_output():
    sys.stdout = sys.__stdout__

def random_culture(n_cultures, culture):
    choices = [i for i in range(n_cultures) if i != culture]  # Exclude `culture`
    return random.choice(choices) if choices else None  # Return None if no valid choices

class ProcessingClass:
    """
    ProcessingClass is a middleware that takes into account
    the the processing modules for testing the models
    """

    def __init__(
        self, shallow, lamp, gpu=False, memory_limit=2700, basePath="./"
    ) -> None:
        """
        init function initialize the dataset object and the gpu setup
        :param shallow: if enabled, shallow learning mode is activated and
        we can use models such as Linear SVM, Gaussian SVM, ... If so,
        we have the images to be greyscale and then flattened, else, we can use
        deep learning algorithms (such as RESNER), so we must have images as RGB
        :param lamp: if enabled we get the images from lamp folder, else from carpet
        folder
        :param gpu: if enabled we use the gpu, else we use the cpu
        """
        if shallow:
            strObj = ShallowStrings()
            if lamp:
                paths = strObj.lamp_paths
            else:
                paths = None
        else:
            strObj = DeepStrings()
            if lamp:
                paths = strObj.lamp_paths
            else:
                paths = strObj.carpet_paths_str
        if paths:
            self.dataobj = DataClass(paths)
        else:
            raise Exception("Carpet Problem has not been tackled in shallow learning")
        self.shallow = shallow
        self.lamp = lamp
        self.basePath = basePath
        if gpu:
            gpus = tf.config.experimental.list_physical_devices("GPU")
            if gpus:
                # Restrict TensorFlow to only allocate 2GB of memory on the first GPU
                try:
                    tf.config.experimental.set_virtual_device_configuration(
                        gpus[0],
                        [
                            tf.config.experimental.VirtualDeviceConfiguration(
                                memory_limit=memory_limit
                            )
                        ],
                    )
                    logical_gpus = tf.config.experimental.list_logical_devices("GPU")
                    print(
                        len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs"
                    )

                except RuntimeError as e:
                    # Virtual devices must be set before GPUs have been initialized
                    print(e)
            else:
                print("no gpus")
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    def parify_batches(self, s, culture, hot_encoding, size):
        #print(f"s is {s}")
        #print(f"s[0] is {s[0]}")
        #print(f"s[1] is {s[1]}")
        indeces_per_culture = []
        batch_size = 64
        if hot_encoding:
            for i in range(self.n_cultures):
                c = np.zeros(self.n_cultures)
                c[i] = 1.0
               
                vals = (np.where((np.asarray(s[1], dtype=object)[:, :self.n_cultures] == c).all(axis=1))[0])
                if i == culture:
                    n_samples_majority = len(vals)
                indeces_per_culture.append(np.where((np.asarray(s[1], dtype=object)[:, :self.n_cultures] == c).all(axis=1))[0])
        else:
            n_samples_majority = len(np.where(np.asarray(s[1], dtype=object)[self.n_cultures]==culture))
            for i in range(self.n_cultures):
                indeces_per_culture.append([np.where(np.asarray(s[1], dtype=object)[:,1])==i])

        B = []
        Blabel = []
        DS = []
        DSlabel = []
        for i in range(n_samples_majority):
            for j in range(self.n_cultures):
                if len(B)>= batch_size:
                    DS.append(np.asarray(B))
                    B = []
                    DSlabel.append(np.asarray(Blabel))
                    Blabel = []

                sample = np.asarray(s[0])[indeces_per_culture[j][i % len(indeces_per_culture[j])]]
                label = np.asarray(s[1])[indeces_per_culture[j][i % len(indeces_per_culture[j])]]
                B.append(cv2.resize(sample, (size, size), interpolation = cv2.INTER_CUBIC))
                Blabel.append(label)

        if len(B)>0:
            for i in range(batch_size-len(B)):
                j = i % self.n_cultures
                rnd_index = np.random.randint(0, len(indeces_per_culture[j]))
                B.append(s[0][indeces_per_culture[j][rnd_index]])
                Blabel.append(s[1][indeces_per_culture[j][rnd_index]])
            DS.append(np.asarray(B))
            DSlabel.append(np.asarray(Blabel))

        return DS, DSlabel

    def prepare_data(
        self,
        standard,
        culture,
        percent=0,
        val_split: float = 0.2,
        test_split: float = 0.2,
        n: int = 1000,
        augment=0,
        gaug = 0.01,
        adversarial=0,
        imbalanced=0,
        discriminator=0,
        diffusion = 0,
        aug = 0,
        weights = 0,
        only_minority_diffusion=0,
        parify_batches_diffusion=0
    ):
        """
        This function prepares the data for training

        :param standard: if enabled, we prepare the dataset for
        standard ML, else our mitigation strategy
        :param culture: culture is an integer number from 0 to |C|-1,
        that represents the majority culture used for training the dataset
        :param percent: is the percentage of images from their dataset of the minority cultures
        :param val_split: is the proportion of the Validation Set w.r.t the union of the Learning and Validation sets
        :param test_split: is the proprtion of the Test Set w.r.t the whole dataset
        :param n: is the maximum number of images contained in each cultural dataset for each class
        :param augment: if enabled, we augment the dataset
        :param g_rot: if augment is enabled, is the gain of random rotation
        :param g_noise: if augment is enabled, is the gain of gaussian noise
        :param g_bright: if augment is enabled, is the gain of random brightness
        """
        self.dataobj.prepare(
            standard=standard,
            culture=culture,
            percent=percent,
            shallow=self.shallow,
            val_split=val_split,
            test_split=test_split,
            n=n,
            adversarial=adversarial or discriminator,
            imbalanced=imbalanced,

        )
        if augment:
            print("Training Augmentation...")
            suppress_output()
            prepObj = PreprocessingClass()
            X_augmented = prepObj.classical_augmentation(
                X=self.dataobj.X, g=gaug, 
            )
        
            self.dataobj.X.extend(X_augmented)
            self.dataobj.y.extend(self.dataobj.y)
            restore_output()
            del X_augmented
            del prepObj 
        
        if diffusion==1 and not discriminator:
            print(f"Diffusion")
            size = 100
            n_imgs = len(self.dataobj.X)//5
            diff_model = DiffusionStandardModel(image_size=size)
            init_shape = np.shape(self.dataobj.X[0])[0:2]

            bpath = "./"
            if standard==0:
                bpath = bpath + '/MIT/'
            if parify_batches_diffusion:
                bpath = bpath + '/PAR_BS/'
            if only_minority_diffusion:
                bpath = bpath + '/ONLY_MIN/'

            fObj = FileManagerClass(bpath)
            del fObj

            fObj = FileManagerClass(bpath+'/GeneratedImages/')
            del fObj

            if parify_batches_diffusion:
                # TODO: modify this
                ########################################################################################
                    if standard and (not adversarial):
                            for j in range(2):
                                tempX = [
                                    cv2.resize(self.dataobj.X[i], (size, size), interpolation = cv2.INTER_CUBIC)
                                    for i in range(len(self.dataobj.X))
                                    if self.dataobj.y[i]== j
                                ]
                                tempXv = [
                                    cv2.resize(self.dataobj.Xv[i], (size, size), interpolation = cv2.INTER_CUBIC)
                                    for i in range(len(self.dataobj.Xv))
                                    if self.dataobj.yv[i] == j
                                ]
                                tempY = [
                                    self.dataobj.y[i]
                                    for i in range(len(self.dataobj.X))
                                    if self.dataobj.y[i]== j
                                ]
                                tempYv = [
                                    self.dataobj.yv[i]
                                    for i in range(len(self.dataobj.Xv))
                                    if self.dataobj.yv[i] == j
                                ]
                                print(tempX)
                                tempX, _ = self.parify_batches((tempX, tempY), culture, False, size)
                                tempXv, _ = self.parify_batches((tempXv, tempYv), culture, False, size)
                                images = diff_model.learn_on_custom_dataset(tempX, tempXv, n_images = n_imgs, plot_imgs = True, aug=aug, percent=percent, lamp=self.lamp, culture=culture, category=j, imb=imbalanced, parify_batches_diffusion=parify_batches_diffusion, base_path=bpath)
                                for img in images:
                                    img = np.asarray(img)
                                    img = cv2.resize(img,  init_shape, interpolation = cv2.INTER_CUBIC)
                                    img = np.asarray(img, dtype=np.float32)
                                    self.dataobj.X.append(img)
                                    self.dataobj.y.append(j)

                    else:
                        for j in range(2):
                            
                            tempX = [
                                cv2.resize(self.dataobj.X[i], (size, size), interpolation = cv2.INTER_CUBIC)
                                for i in range(len(self.dataobj.X))
                                if self.dataobj.y[i][self.n_cultures]== j
                            ]
                            tempXv = [
                                cv2.resize(self.dataobj.Xv[i], (size, size), interpolation = cv2.INTER_CUBIC)
                                for i in range(len(self.dataobj.Xv))
                                if self.dataobj.yv[i][self.n_cultures] == j
                            ]    
                            tempY = [
                                self.dataobj.y[i]
                                for i in range(len(self.dataobj.X))
                                if self.dataobj.y[i][self.n_cultures]== j
                            ]
                            tempYv = [
                                self.dataobj.yv[i]
                                for i in range(len(self.dataobj.Xv))
                                if self.dataobj.yv[i][self.n_cultures] == j
                            ]     
                            tempX, _ = self.parify_batches((tempX, tempY), culture, True, size)
                            tempXv, _ = self.parify_batches((tempXv, tempYv), culture, True, size)               
                            images = diff_model.learn_on_custom_dataset(tempX, tempXv, n_images = n_imgs, plot_imgs = True, aug=aug, percent=percent, lamp=self.lamp, culture=culture, category=j, imb=imbalanced, parify_batches_diffusion=parify_batches_diffusion,  base_path=bpath)
                            for img in images:
                                img = np.asarray(img)
                                img = cv2.resize(img,  init_shape, interpolation = cv2.INTER_CUBIC)
                                img = np.asarray(img, dtype=np.float32)
                                self.dataobj.X.append(img)
                                c = random_culture(self.n_cultures, culture)
                                lbl = list(np.zeros(self.n_cultures)) 
                                lbl[c]=1.0
                                lbl.append(j)
                                self.dataobj.y.append(lbl)
            else: 
                if only_minority_diffusion:
                    if standard and (not adversarial):
                        if imbalanced:
                            for j in range(2):
                                tempX = []
                                tempXv = []
                                for i in range(len(self.dataobj.X)):
                                    if self.dataobj.y[i][1]== j and self.dataobj.y[i][0]==culture:
                                        for i in range(int(1/weights[self.dataobj.y[i][0]])): # I use the inverse of the total proportion for augmenting the dataset
                                            tempX.append(cv2.resize(self.dataobj.X[i], (size, size), interpolation = cv2.INTER_CUBIC)) 
                                for i in range(len(self.dataobj.Xv)):
                                    if self.dataobj.yv[i][1]== j and self.dataobj.y[i][0]==culture:
                                        for i in range(int(1/weights[self.dataobj.y[i][0]])): # I use the inverse of the total proportion for augmenting the dataset
                                            tempXv.append(cv2.resize(self.dataobj.Xv[i], (size, size), interpolation = cv2.INTER_CUBIC)) 
                            
                                images = diff_model.learn_on_custom_dataset(tempX, tempXv, n_images = n_imgs, plot_imgs = True, aug=aug, percent=percent, lamp=self.lamp, culture=culture, category=j, imb=imbalanced, parify_batches_diffusion=parify_batches_diffusion,  base_path=bpath)
                                for img in images:
                                    img = np.asarray(img)
                                    img = cv2.resize(img,  init_shape, interpolation = cv2.INTER_CUBIC)
                                    img = np.asarray(img, dtype=np.float32)
                                    self.dataobj.X.append(img)
                                    c = random_culture(self.n_cultures, culture)
                                    self.dataobj.y.append([c, j]) # I do not invent a new culture anymore as I need to use the information in mitigation

                        else:
                            for j in range(2):
                                tempX = [
                                    cv2.resize(self.dataobj.X[i], (size, size), interpolation = cv2.INTER_CUBIC)
                                    for i in range(len(self.dataobj.X))
                                    if self.dataobj.y[i]== j and self.dataobj.y[i][0]==culture
                                ]
                                tempXv = [
                                    cv2.resize(self.dataobj.Xv[i], (size, size), interpolation = cv2.INTER_CUBIC)
                                    for i in range(len(self.dataobj.Xv))
                                    if self.dataobj.yv[i] == j and self.dataobj.y[i][0]==culture
                                ]
                                images = diff_model.learn_on_custom_dataset(tempX, tempXv, n_images = n_imgs, plot_imgs = True, aug=aug, percent=percent, lamp=self.lamp, culture=culture, category=j, imb=imbalanced, parify_batches_diffusion=parify_batches_diffusion,  base_path=bpath)
                                for img in images:
                                    img = np.asarray(img)
                                    img = cv2.resize(img,  init_shape, interpolation = cv2.INTER_CUBIC)
                                    img = np.asarray(img, dtype=np.float32)
                                    self.dataobj.X.append(img)
                                    self.dataobj.y.append(j)

                    else:
                        for j in range(2):
                            
                            tempX = [
                                cv2.resize(self.dataobj.X[i], (size, size), interpolation = cv2.INTER_CUBIC)
                                for i in range(len(self.dataobj.X))
                                if self.dataobj.y[i][self.n_cultures]== j and np.argmax(self.dataobj.y[i][0:self.n_cultures])==culture
                            ]
                            tempXv = [
                                cv2.resize(self.dataobj.Xv[i], (size, size), interpolation = cv2.INTER_CUBIC)
                                for i in range(len(self.dataobj.Xv))
                                if self.dataobj.yv[i][self.n_cultures] == j and np.argmax(self.dataobj.y[i][0:self.n_cultures])==culture
                            ]                    
                            images = diff_model.learn_on_custom_dataset(tempX, tempXv, n_images = n_imgs, plot_imgs = True, aug=aug, percent=percent, lamp=self.lamp, culture=culture, category=j, imb=imbalanced, parify_batches_diffusion=parify_batches_diffusion, base_path=bpath)
                            for img in images:
                                img = np.asarray(img)
                                img = cv2.resize(img,  init_shape, interpolation = cv2.INTER_CUBIC)
                                img = np.asarray(img, dtype=np.float32)
                                self.dataobj.X.append(img)
                                c = random_culture(self.n_cultures, culture)
                                lbl = list(np.zeros(self.n_cultures)) # Generated images are equidistant from the cultures
                                lbl[c]=1.0
                                lbl.append(j)
                                self.dataobj.y.append(lbl)
                    
                else:
                    if standard and (not adversarial):
                        if imbalanced:
                            for j in range(2):
                                tempX = []
                                tempXv = []
                                for i in range(len(self.dataobj.X)):
                                    if self.dataobj.y[i][1]== j:
                                        for i in range(int(1/weights[self.dataobj.y[i][0]])): # I use the inverse of the total proportion for augmenting the dataset
                                            tempX.append(cv2.resize(self.dataobj.X[i], (size, size), interpolation = cv2.INTER_CUBIC)) 
                                for i in range(len(self.dataobj.Xv)):
                                    if self.dataobj.yv[i][1]== j:
                                        for i in range(int(1/weights[self.dataobj.y[i][0]])): # I use the inverse of the total proportion for augmenting the dataset
                                            tempXv.append(cv2.resize(self.dataobj.Xv[i], (size, size), interpolation = cv2.INTER_CUBIC)) 
                            
                                images = diff_model.learn_on_custom_dataset(tempX, tempXv, n_images = n_imgs, plot_imgs = True, aug=aug, percent=percent, lamp=self.lamp, culture=culture, category=j, imb=imbalanced, parify_batches_diffusion=parify_batches_diffusion, base_path=bpath)
                                for img in images:
                                    img = np.asarray(img)
                                    img = cv2.resize(img,  init_shape, interpolation = cv2.INTER_CUBIC)
                                    img = np.asarray(img, dtype=np.float32)
                                    self.dataobj.X.append(img)
                                    c = random_culture(self.n_cultures, culture)
                                    self.dataobj.y.append([c, j]) # I have to invent another culture

                        else:
                            for j in range(2):
                                tempX = [
                                    cv2.resize(self.dataobj.X[i], (size, size), interpolation = cv2.INTER_CUBIC)
                                    for i in range(len(self.dataobj.X))
                                    if self.dataobj.y[i]== j
                                ]
                                tempXv = [
                                    cv2.resize(self.dataobj.Xv[i], (size, size), interpolation = cv2.INTER_CUBIC)
                                    for i in range(len(self.dataobj.Xv))
                                    if self.dataobj.yv[i] == j
                                ]
                                images = diff_model.learn_on_custom_dataset(tempX, tempXv, n_images = n_imgs, plot_imgs = True, aug=aug, percent=percent, lamp=self.lamp, culture=culture, category=j, imb=imbalanced, parify_batches_diffusion=parify_batches_diffusion, base_path=bpath)
                                for img in images:
                                    img = np.asarray(img)
                                    img = cv2.resize(img,  init_shape, interpolation = cv2.INTER_CUBIC)
                                    img = np.asarray(img, dtype=np.float32)
                                    self.dataobj.X.append(img)
                                    self.dataobj.y.append(j)

                    else:
                        for j in range(2):
                            print(f"second cycle")
                            for i in range(len(self.dataobj.X)):
                                print(self.dataobj.y[i])
                            tempX = [
                                cv2.resize(self.dataobj.X[i], (size, size), interpolation = cv2.INTER_CUBIC)
                                for i in range(len(self.dataobj.X))
                                if self.dataobj.y[i][self.n_cultures]== j
                            ]
                            tempXv = [
                                cv2.resize(self.dataobj.Xv[i], (size, size), interpolation = cv2.INTER_CUBIC)
                                for i in range(len(self.dataobj.Xv))
                                if self.dataobj.yv[i][self.n_cultures] == j
                            ]                    
                            images = diff_model.learn_on_custom_dataset(tempX, tempXv, n_images = n_imgs, plot_imgs = True, aug=aug, percent=percent, lamp=self.lamp, culture=culture, category=j, imb=imbalanced, parify_batches_diffusion=parify_batches_diffusion, base_path=bpath)
                            for img in images:
                                img = np.asarray(img)
                                img = cv2.resize(img,  init_shape, interpolation = cv2.INTER_CUBIC)
                                img = np.asarray(img, dtype=np.float32)
                                self.dataobj.X.append(img)
                                c = random_culture(self.n_cultures, culture)
                                lbl = list(np.zeros(self.n_cultures)) 
                                lbl[c]=1.0
                                lbl.append(j)
                                self.dataobj.y.append(lbl)
                    
            del diff_model   
        
        

    def prepare_test(
        self,
        augment=0,
        g_rot: float = 0.1,
        g_noise: float = 0.1,
        g_bright: float = 0.1,
        adversary=0,
        culture=None,
        eps=0.3,
        nt=None,
    ):
        """
        This function prepares the data for testing

        :param augment: if enabled, we augment the dataset
        :param g_rot: if augment is enabled, is the gain of random rotation
        :param g_noise: if augment is enabled, is the gain of gaussian noise
        :param g_bright: if augment is enabled, is the gain of random brightness
        :param adversary: if enabled, we augment the dataset using adversary samples
        :param culture: if adversary is enabled, we need the output information for implementing
        fast gradient method
        :param eps: is adversary is enabled, it is the gain of fast gradient method
        :param nt: is the number of images to use for testing

        """
        self.Xt_totaug = []
        self.Xt_adv = []
        self.Xt_aug = []
        if nt != None and nt < len(self.dataobj.Xt):
            self.dataobj.Xt = self.dataobj.Xt[0:nt]
        for culture in range(3):
            if augment:
                if adversary:
                    if self.model != None and culture != None:
                            print("Preparing Tot Aug for Testing...")
                            prepObj = PreprocessingClass()
                            Xt_aug = prepObj.classical_augmentation(
                                X=self.dataobj.Xt[culture],
                                g_rot=g_rot,
                                g_noise=g_noise,
                                g_bright=g_bright,
                            )
                            self.Xt_totaug.append(
                                prepObj.adversarial_augmentation(
                                    X=Xt_aug,
                                    y=self.dataobj.yt[culture],
                                    model=self.model,
                                    culture=culture,
                                    eps=eps,
                                )
                            )
                            del prepObj
                    else:
                        raise Exception(
                            "Incorrect call for prepare_test, missing model or culture"
                        )
                else:
                    
                        print("Preparing Aug for Testing...")
                        prepObj = PreprocessingClass()
                        self.Xt_aug.append(
                            prepObj.classical_augmentation(
                                X=self.dataobj.Xt[culture],
                                g_rot=g_rot,
                                g_noise=g_noise,
                                g_bright=g_bright,
                            )
                        )
                        del prepObj
            else:
                if adversary:
                    if self.model != None and culture != None:
                        print("Preparing Adv for Testing...")
                        
                        prepObj = PreprocessingClass()
                        self.Xt_adv.append(
                            prepObj.adversarial_augmentation(
                                X=self.dataobj.Xt[culture],
                                y=self.dataobj.yt[culture],
                                model=self.model,
                                culture=culture,
                                eps=eps,
                            )
                        )
                        del prepObj
                    else:
                        raise Exception(
                            "Incorrect call for prepare_test, missing model or culture"
                        )

    def process(
        self,
        standard,
        type="DL",
        points=50,
        kernel="linear",
        verbose_param=0,
        learning_rate=0.001,
        epochs=15,
        batch_size=2,
        lambda_index=0,
        culture=0,
        percent=0,
        val_split: float = 0.2,
        test_split: float = 0.1,
        n: int = 1000,
        augment=0,
        gaug: float = 0.1,
        discriminator=0,
        adversary=0,
        eps=0.3,
        gradcam=False,
        complete=0,
        n_cultures=3,
        imbalanced=0,
        class_division=0,
        only_imb_imgs=0,
        diffusion=0,
        only_minority_diffusion=0,
        parify_batches_diffusion=0,
        mitigation_type=0
    ):
        """
        process function prepares the data and fit the model

        This function prepares the data for training

        :param standard: if enabled, we prepare the dataset for
        standard ML, else our mitigation strategy
        :param type: select the algorithm, possible values: (SVM and DL/RESNET)
        :param points: if the selected algorithm is SVM, this value sets the number of points used in the grid
        :param kernel: if the selected algorithm is SVM, this value sets the kernel (linear or gaussian)
        :param verbose_param: sets the verbose mode
        :param learning_rate: if the selected algorithm is DL, this value sets the gain of the step
        :param epochs: if the selected algorithm is DL, this value sets the number of epochs
        :param lambda_index: if we are in our Mitigation Strategy mode, it selectes the gain of the regularizer
        :param batchs_size: if the selected algorithm is DL, this value sets the batch size
        :param culture: culture is an integer number from 0 to |C|-1,
        that represents the majority culture used for training the dataset
        :param percent: is the percentage of images from their dataset of the minority cultures
        :param val_split: is the proportion of the Validation Set w.r.t the union of the Learning and Validation sets
        :param test_split: is the proprtion of the Test Set w.r.t the whole dataset
        :param n: is the maximum number of images contained in each cultural dataset for each class
        :param augment: if enabled, we augment the dataset
        :param g_rot: if augment is enabled, is the gain of random rotation
        :param g_noise: if augment is enabled, is the gain of gaussian noise
        :param g_bright: if augment is enabled, is the gain of random brightness
        :param culture: if adversary is enabled, we need the output information for implementing
        fast gradient method
        :param eps: is adversary is enabled, it is the gain of fast gradient method
        :param nt: is the number of images to use for testing
        :param gradcam: if enabled, we extrapolate the GradCAM during training for explainability
        """
        weights = np.ones(n_cultures) * 1/3 #percent
        weights[culture] = 1  # this are the proportions in the dataset
        self.n_cultures = n_cultures
        self.prepare_data(
            standard=standard,
            culture=culture,
            percent=percent,
            val_split=val_split,
            test_split=test_split,
            n=n,
            augment=augment,
            adversarial=adversary,
            imbalanced=imbalanced,
            discriminator=discriminator,
            diffusion = diffusion,
            gaug = gaug,
            weights = weights,
            only_minority_diffusion=only_minority_diffusion,
            parify_batches_diffusion=parify_batches_diffusion
        )
        self.model = None
        
        # Base path:
        # - STD/MIT
        # - model: SVC, RFC, DL
        # - culture: LC, LF, LT, CI, CJ, CS
        # - augment in TS: NOAUG, STDAUG, ADV, TOTAUG
        # - lambda index: -1, 0, 1, ...
        # Complete path:
        # - augment in Test: TNOAUG, TSTDAUG, TADV, TTOTAUG
        if discriminator:
            self.basePath = self.basePath + "/DISCR/STD/" + type
        else:
            if standard:
                self.basePath = self.basePath + "STD/" + type
            else:
                self.basePath = self.basePath + "MIT/" + type

        if imbalanced:
            self.basePath = self.basePath + "/IMB/"
        else:
            self.basePath = self.basePath + "/BAL/"
        if self.lamp:
            if culture == 0:
                c = "/LC/"
            elif culture == 1:
                c = "/LF/"
            elif culture == 2:
                c = "/LT/"
            else:
                c = "/LC/"
        else:
            if culture == 0:
                c = "/CI/"
            elif culture == 1:
                c = "/CJ/"
            elif culture == 2:
                c = "/CS/"
            else:
                c = "/CI/"
        self.basePath = self.basePath + c + str(percent) + "/"
        if diffusion: 
            self.basePath = self.basePath + "DIFFUSION/"
            if only_minority_diffusion:
                self.basePath = self.basePath + "ONLY_MIN/"
        if parify_batches_diffusion:
                self.basePath = self.basePath + "PAR_BS/"
        if augment:
            if adversary:
                if only_imb_imgs:
                    aug = f"ADD_TOTAUG/g={gaug}/eps={eps}/"
                else:
                    aug = f"TOTAUG/g={gaug}/eps={eps}/"
                if class_division:
                    aug = aug + "/CLSDIV/"
                else:
                    aug = aug + "/NOCLSDIV/"
           

            else:
                aug = f"STDAUG/g={gaug}/"
        else:
            if adversary :
                if only_imb_imgs:
                    aug = f"ADD_AVD/eps={eps}/"
                else:
                    aug = f"AVD/eps={eps}/"
                if class_division:
                    aug = aug + "/CLSDIV/"
                else:
                    aug = aug + "/NOCLSDIV/"
            
            else:
                aug = "NOAUG/"

        self.basePath = self.basePath + aug
        if (not standard) and (not complete):
            self.basePath = self.basePath + str(lambda_index) + "/"

        if discriminator:
            self.model = Discriminator(
                type=type,
                points=points,
                kernel=kernel,
                verbose_param=verbose_param,
                learning_rate=learning_rate,
                epochs=epochs,
                batch_size=batch_size,
                weights=weights,
                imbalanced=imbalanced,
                class_division=class_division,
                
            )
        else:
            if standard:
                if adversary:
                    self.model = AdversarialStandard(
                        type=type,
                        points=points,
                        kernel=kernel,
                        verbose_param=verbose_param,
                        learning_rate=learning_rate,
                        epochs=epochs,
                        batch_size=batch_size,
                        weights=weights,
                        imbalanced=imbalanced,
                        class_division=class_division,
                        only_imb_imgs=only_imb_imgs,
                        path = self.basePath,
                    )
                    
                else:
                    if gradcam:
                        self.model = StandardModels4GradCam(
                            type=type,
                            points=points,
                            kernel=kernel,
                            verbose_param=verbose_param,
                            learning_rate=learning_rate,
                            epochs=epochs,
                            batch_size=batch_size,
                            weights=weights,
                            imbalanced=imbalanced,
                            diffusion=diffusion,
                        )
                    else:
                        self.model = StandardModels(
                            type=type,
                            points=points,
                            kernel=kernel,
                            verbose_param=verbose_param,
                            learning_rate=learning_rate,
                            epochs=epochs,
                            batch_size=batch_size,
                            weights=weights,
                            imbalanced=imbalanced,
                            diffusion=diffusion,    
                            path = self.basePath,                        
                        )
            else:
                if mitigation_type==1:
                    self.model = MitigatedModelsAdvanced(
                        type=type,
                        culture=culture,
                        verbose_param=verbose_param,
                        epochs=epochs,
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                        lambda_index=lambda_index,
                        n_cultures=n_cultures,
                        imbalanced=imbalanced,
                        diffusion=diffusion,
                        weights=weights,
                        parify_batches_diffusion=parify_batches_diffusion
                    )
                else:   
                    self.model = MitigatedModels(
                        type=type,
                        culture=culture,
                        verbose_param=verbose_param,
                        epochs=epochs,
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                        lambda_index=lambda_index,
                        n_cultures=n_cultures,
                        imbalanced=imbalanced,
                        diffusion=diffusion,
                        weights=weights,
                        parify_batches_diffusion=parify_batches_diffusion
                    )

        self.model.standard = standard
        self.model.fit(
            (self.dataobj.X, self.dataobj.y),
            (self.dataobj.Xv, self.dataobj.yv),
            eps=eps,
            gradcam=gradcam,
            out_dir=self.basePath,
            complete=complete,
            aug=augment,
            g=gaug,
        )
        if adversary:
            self.prepare_test()
            if class_division:
                for j in range(2):
                    self.discriminator_test( augment, imbalanced, self.model.adversarial_model[j], j)
            else:
                self.discriminator_test( augment, imbalanced, self.model.adversarial_model)
        self.imbalanced = imbalanced
        
        del c
        del aug


    def discriminator_test(self, augment, imbalanced, model, j=-1):
        discriminator_model = Discriminator(imbalanced=imbalanced)
        discriminator_model.model = model
        if augment:
                cm = discriminator_model.get_model_stats(
                    self.Xt_aug, self.dataobj.yt, discriminator=1
                )
                testaug = f"TSTDAUG/G_AUG={gaug}/"
        else:
                cm = discriminator_model.get_model_stats(
                    self.dataobj.Xt, self.dataobj.yt, discriminator=1
                )
                testaug = f"TNOAUG/"
        testaug = testaug + f"CULTURE/"
        path = self.basePath + testaug + f"res_scrimin={j}.csv"
        self.save_results(cm, path, discriminator=1)

    def test(
        self,
        standard,
        culture=0,
        augment=0,
        gaug=0.1,
        adversary=0,
        eps=0.3,
        nt=None,
        discriminator=0,
    ):
        """
        This function is used for testing the model

        :param augment: if enabled, we augment the dataset
        :param g_rot: if augment is enabled, is the gain of random rotation
        :param g_noise: if augment is enabled, is the gain of gaussian noise
        :param g_bright: if augment is enabled, is the gain of random brightness
        :param adversary: if enabled, we augment the dataset using adversary samples
        :param culture: if adversary is enabled, we need the output information for implementing
        fast gradient method
        :param eps: is adversary is enabled, it is the gain of fast gradient method
        :param nt: is the number of images to use for testing

        :return -1 is the model is not trained, 0 if end the testing phase
        """
        if self.model:
            self.prepare_test(
                augment=augment,
                g_rot=gaug,
                g_noise=gaug,
                g_bright=gaug,
                adversary=adversary,
                culture=culture,
                eps=eps,
            )
        else:
            print("Pay attention: no model information given for tests")
            return -1
        if discriminator==0:
            for culture in range(3):
                if standard:
                    if augment:
                        if adversary:
                            cm = self.model.get_model_stats(
                                self.Xt_totaug[culture],
                                self.dataobj.yt[culture],
                                discriminator=discriminator,
                            )
                            testaug = f"TTOTAUG/G_AUG={gaug}/EPS={eps}/"
                        else:
                            cm = self.model.get_model_stats(
                                self.Xt_aug[culture],
                                self.dataobj.yt[culture],
                                discriminator=discriminator,
                            )
                            testaug = f"TSTDAUG/G_AUG={gaug}/"
                    else:
                        if adversary:
                            cm = self.model.get_model_stats(
                                self.Xt_adv[culture],
                                self.dataobj.yt[culture],
                                discriminator=discriminator,
                            )
                            testaug = f"TAVD/EPS={eps}/"
                        else:
                            cm = self.model.get_model_stats(
                                self.dataobj.Xt[culture],
                                self.dataobj.yt[culture],
                                discriminator=discriminator,
                            )
                            testaug = f"TNOAUG/"
                    testaug = testaug + f"CULTURE{culture}/"
                    path = self.basePath + testaug + "res.csv"
                    self.save_results(cm, path, discriminator=discriminator)
                else:
                    for i in range(3):
                        if augment:
                            if adversary:
                                cm = self.model.get_model_stats(
                                    self.Xt_totaug[culture],
                                    self.dataobj.yt[culture],
                                    i,
                                    discriminator=discriminator,
                                )
                                testaug = f"TTOTAUG/G_AUG={gaug}/EPS={eps}/"
                            else:
                                cm = self.model.get_model_stats(
                                    self.Xt_aug[culture],
                                    self.dataobj.yt[culture],
                                    i,
                                    discriminator=discriminator,
                                )
                                testaug = f"TSTDAUG/G_AUG={gaug}/"
                        else:
                            if adversary:
                                cm = self.model.get_model_stats(
                                    self.Xt_adv[culture],
                                    self.dataobj.yt[culture],
                                    i,
                                    discriminator=discriminator,
                                )
                                testaug = f"TAVD/EPS={eps}/"
                            else:
                                cm = self.model.get_model_stats(
                                    self.dataobj.Xt[culture],
                                    self.dataobj.yt[culture],
                                    i,
                                    discriminator=discriminator,
                                )
                                testaug = f"TNOAUG/"
                        testaug = testaug + f"CULTURE{culture}/"
                        path = self.basePath + testaug + "out " + str(i) + ".csv"
                        print(f"Path is {path}")
                        self.save_results(cm, path, discriminator=discriminator)
                        del path
                        del testaug
        else:
            self.discriminator_test( augment, self.imbalanced, self.model)

        return

    def save_results(self, cm, path, discriminator=0):
        """
        :param cm: is the confusion matrix to be saved
        :param path: is the path in which we want to save the confusion matrix
        """
        print(f"Path is {path}")
        fObj = FileManagerClass(path)
        fObj.writecm(cm, discriminator=discriminator)
        del fObj

    def partial_clear(self, basePath=None):
        """
        Partially clear the space for avoiding memory issues
        """
        tf.keras.backend.clear_session()
        self.model = None
        del self.model
        self.dataobj.clear()
        self.Xt_totaug = None
        del self.Xt_totaug
        self.Xt_adv = None
        del self.Xt_adv
        self.Xt_aug = None
        del self.Xt_aug
        self.basePath = basePath

        gc.collect()
