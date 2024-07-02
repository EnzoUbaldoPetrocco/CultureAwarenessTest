#!/usr/bin/env python
__author__ = "Enzo Ubaldo Petrocco"
import sys

sys.path.insert(1, "../")
from Model.mitigated.mitigated_models import MitigatedModels
from Model.standard.standard_models import StandardModels
from Model.adversarial.adversarial import AdversarialStandard 
from Utils.Data.Data import DataClass
from Utils.FileManager.FileManager import FileManagerClass
from Utils.Results.Results import ResultsClass
from Utils.Data.deep_paths import DeepStrings
from Utils.Data.shallow_paths import ShallowStrings
from Utils.Data.Data import PreprocessingClass
import numpy as np
import tensorflow as tf
import os
import gc

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_asyn"


class ProcessingClass:
    """
    ProcessingClass is a middleware that takes into account
    the the processing modules for testing the models
    """
    def __init__(self, shallow, lamp, gpu=False, memory_limit=2700, basePath='./') -> None:
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

    def prepare_data(
        self,
        standard,
        culture,
        percent=0,
        val_split: float = 0.2,
        test_split: float = 0.2,
        n: int = 1000,
        augment=0,
        g_rot: float = 0.1,
        g_noise: float = 0.1,
        g_bright: float = 0.1,
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
        )
        if augment:
            with tf.device("/gpu:0"):
                print("Training Augmentation...")
                prepObj = PreprocessingClass()
                X_augmented = prepObj.classical_augmentation(
                    X=self.dataobj.X, g_rot=g_rot, g_noise=g_noise, g_bright=g_bright
                )
                Xv_augmented = prepObj.classical_augmentation(
                    X=self.dataobj.Xv, g_rot=g_rot, g_noise=g_noise, g_bright=g_bright
                )

            self.dataobj.X.extend(X_augmented)
            self.dataobj.Xv.extend(Xv_augmented)
            self.dataobj.y.extend(self.dataobj.y)
            self.dataobj.yv.extend(self.dataobj.yv)
            del X_augmented
            del Xv_augmented
            del prepObj

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
                        with tf.device("/gpu:0"):
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
                    with tf.device("/gpu:0"):
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
                        with tf.device("/gpu:0"):
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
        complete = 0,
        n_cultures=3
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
        self.prepare_data(
            standard=standard,
            culture=culture,
            percent=percent,
            val_split=val_split,
            test_split=test_split,
            n=n,
            augment=0
        )
        self.model = None
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
                n_cultures=n_cultures
            )

        self.model.standard=standard
            # Base path:
        # - STD/MIT
        # - model: SVC, RFC, DL
        # - culture: LC, LF, LT, CI, CJ, CS
        # - augment in TS: NOAUG, STDAUG, ADV, TOTAUG
        # - lambda index: -1, 0, 1, ...
        # Complete path:
        # - augment in Test: TNOAUG, TSTDAUG, TADV, TTOTAUG
        if standard:
            self.basePath = self.basePath + "STD/" + type
        else:
            self.basePath = self.basePath + "MIT/" + type
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
        for culture in range(3):
            if standard:
                if augment:
                    if adversary:
                        cm = self.model.get_model_stats(
                            self.Xt_totaug[culture], self.dataobj.yt[culture]
                        )
                        testaug = f"TTOTAUG/G_AUG={gaug}/EPS={eps}/"
                    else:
                        cm = self.model.get_model_stats(
                            self.Xt_aug[culture], self.dataobj.yt[culture]
                        )
                        testaug = f"TSTDAUG/G_AUG={gaug}/"
                else:
                    if adversary:
                        cm = self.model.get_model_stats(
                            self.Xt_adv[culture], self.dataobj.yt[culture]
                        )
                        testaug = f"TAVD/EPS={eps}/"
                    else:
                        cm = self.model.get_model_stats(
                            self.dataobj.Xt[culture], self.dataobj.yt[culture]
                        )
                        testaug = f"TNOAUG/"
                testaug = testaug + f"CULTURE{culture}/"
                path = self.basePath + testaug + "res.csv"
                self.save_results(cm, path)
            else:
                for i in range(3):
                    if augment:
                        if adversary:
                            cm = self.model.get_model_stats(
                                self.Xt_totaug[culture], self.dataobj.yt[culture], i
                            )
                            testaug = f"TTOTAUG/G_AUG={gaug}/EPS={eps}/"
                        else:
                            cm = self.model.get_model_stats(
                                self.Xt_aug[culture], self.dataobj.yt[culture], i
                            )
                            testaug = f"TSTDAUG/G_AUG={gaug}/"
                    else:
                        if adversary:
                            cm = self.model.get_model_stats(
                                self.Xt_adv[culture], self.dataobj.yt[culture], i
                            )
                            testaug = f"TAVD/EPS={eps}/"
                        else:
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
