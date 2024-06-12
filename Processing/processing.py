#!/usr/bin/env python
__author__ = "Enzo Ubaldo Petrocco"
import sys

sys.path.insert(1, "../")
from Model.mitigated.mitigated_models import MitigatedModels
from Model.standard.standard_models import StandardModels
from Utils.Data.Data import DataClass
from Utils.FileManager.FileManager import FileManagerClass
from Utils.Data.deep_paths import DeepStrings
from Utils.Data.shallow_paths import ShallowStrings
import numpy as np
import os
import gc
import torch

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class ProcessingClass:
    """
    ProcessingClass is a middleware that takes into account
    the the processing modules for testing the models
    """
    def __init__(self, shallow, lamp, gpu=False, memory_limit=2700) -> None:
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
        if gpu:
            torch.cuda.set_per_process_memory_fraction(0.9, 0)
            torch.cuda.empty_cache()
            total_memory = torch.cuda.get_device_properties(0).total_memory
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    def process(
        self,
        standard,
        type="DL",
        points=50,
        kernel="linear",
        verbose_param=0,
        learning_rate=0.001,
        epochs=15,
        batch_size=15,
        lambda_index=-1,
        culture=0,
        percent=0,
        val_split: float = 0.2,
        test_split: float = 0.2,
        n: int = 1000,
        augment=0,
        gaug: float = 0.1,
        adversary=0,
        eps=0.3,
        mult=0.05,
        gradcam=False,
        complete = 0
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
        self.dataobj.prepare(
            standard=standard,
            culture=culture,
            percent=percent,
            shallow=self.shallow,
            val_split=val_split,
            test_split=test_split,
            n=n,
        )
        self.model = None
        if standard:
            self.model = StandardModels(
                type=type,
                points=points,
                kernel=kernel,
                verbose_param=verbose_param,
                learning_rate=learning_rate,
                epochs=epochs,
                batch_size=batch_size,
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
            )
            # Base path:
        # - STD/MIT
        # - model: SVC, RFC, DL
        # - culture: LC, LF, LT, CI, CJ, CS
        # - augment in TS: NOAUG, STDAUG, ADV, TOTAUG
        # - lambda index: -1, 0, 1, ...
        # Complete path:
        # - augment in Test: TNOAUG, TSTDAUG, TADV, TTOTAUG
        if standard:
            self.basePath = "./STD/" + type
        else:
            self.basePath = "./MIT/" + type
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
        if augment:
            if adversary:
                aug = f"TOTAUG/g={gaug}/"
            else:
                aug = f"STDAUG/g={gaug}/"
        else:
            if adversary:
                aug = "AVD/"
            else:
                aug = "NOAUG/"

        self.basePath = self.basePath + aug
        if ((not standard) and (not complete)):
            self.basePath = self.basePath + str(lambda_index) + "/"
        del c
        del aug
        self.model.fit(
            (self.dataobj.X, self.dataobj.y),
            (self.dataobj.Xv, self.dataobj.yv),
            adversary=adversary,
            eps=eps,
            mult=mult,
            gradcam=gradcam,
            out_dir=self.basePath,
            complete = complete,
            aug=augment,
            g=gaug
        )

    def test(
        self,
        standard,
        culture=0,
        augment=0,
        gaug=0.1,
        adversary=0,
        eps=0.3,
        nt = None
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
        
        for culture in range(3):
            if standard:
                cm = self.model.get_model_stats(
                    self.dataobj.Xt[culture], self.dataobj.yt[culture]
                )
                testaug = f"TNOAUG/"
                testaug = testaug + f"CULTURE{culture}/"
                path = self.basePath + testaug + "res.csv"
                self.save_results(cm, path)
            else:
                for i in range(3):
                    
                    cm = self.model.get_model_stats(
                        self.dataobj.Xt[culture], self.dataobj.yt[culture], i
                    )
                    testaug = f"TNOAUG/"
                    testaug = testaug + f"CULTURE{culture}/"
                    path = self.basePath + testaug + "out " + str(i) + ".csv"
                    self.save_results(cm, path)
                    del path
                    del testaug
        return 0

    def save_results(self, cm, path):
        """
        :param cm: is the confusion matrix to be saved
        :param path: is the path in which we want to save the confusion matrix
        """
        fObj = FileManagerClass(path)
        fObj.writecm(cm)
        del fObj

    def partial_clear(self):
        """
        Partially clear the space for avoiding memory issues
        """
        self.model = None
        del self.model
        self.dataobj.clear()
        self.Xt_totaug = None
        del self.Xt_totaug
        self.Xt_adv = None
        del self.Xt_adv
        self.Xt_aug = None
        del self.Xt_aug
        self.basePath = None
        del self.basePath
        gc.collect()
