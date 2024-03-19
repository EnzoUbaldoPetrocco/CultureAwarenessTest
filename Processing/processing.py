import sys

sys.path.insert(1, "../")
from Model.mitigated.mitigated_models import MitigatedModels
from Model.standard.standard_models import StandardModels
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


class ProcessingClass:
    def __init__(self, shallow, lamp, gpu=False) -> None:

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
            gpus = tf.config.experimental.list_physical_devices("GPU")
            if gpus:
                # Restrict TensorFlow to only allocate 2GB of memory on the first GPU
                try:
                    tf.config.experimental.set_virtual_device_configuration(
                        gpus[0],
                        [
                            tf.config.experimental.VirtualDeviceConfiguration(
                                memory_limit=10000
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
        batch_size=15,
    ):
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
        nt = None
    ):
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
        g_rot: float = 0.1,
        g_noise: float = 0.1,
        g_bright: float = 0.1,
        adversary=0,
        eps=0.3,
        mult=0.05,
        gradcam = False
    ):

        self.prepare_data(
            standard=standard,
            culture=culture,
            percent=percent,
            val_split=val_split,
            test_split=test_split,
            n=n,
            augment=augment,
            g_rot=g_rot,
            g_noise=g_noise,
            g_bright=g_bright,
            batch_size=batch_size,
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
                lambda_index=lambda_index,
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
                aug = "TOTAUG/"
            else:
                aug = "STDAUG/"
        else:
            if adversary:
                aug = "AVD/"
            else:
                aug = "NOAUG/"
        
        self.basePath = self.basePath + aug
        if not standard:
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
            out_dir=self.basePath
        )


    def test(
        self,
        standard,
        culture=0,
        augment=0,
        g_rot: float = 0.1,
        g_noise: float = 0.1,
        g_bright: float = 0.1,
        adversary=0,
        eps=0.3,
    ):
        if self.model:
            self.prepare_test(
                augment=augment,
                g_rot=g_rot,
                g_noise=g_noise,
                g_bright=g_bright,
                adversary=adversary,
                culture=culture,
                eps=eps,
            )
        else:
            print("Pay attention: no model information given for tests")
        for culture in range(3):
            if standard:
                if augment:
                    if adversary:
                        cm = self.model.get_model_stats(
                            self.Xt_totaug[culture], self.dataobj.yt[culture]
                        )
                        testaug = f"TTOTAUG/G_AUG={g_noise}/EPS={eps}/"
                    else:
                        cm = self.model.get_model_stats(
                            self.Xt_aug[culture], self.dataobj.yt[culture]
                        )
                        testaug = f"TSTDAUG/G_AUG={g_noise}/"
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
                            testaug = f"TTOTAUG/G_AUG={g_noise}/EPS={eps}/"
                        else:
                            cm = self.model.get_model_stats(
                                self.Xt_aug[culture], self.dataobj.yt[culture], i
                            )
                            testaug = f"TSTDAUG/G_AUG={g_noise}/"
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

    def save_results(self, cm, path):
        fObj = FileManagerClass(path)
        fObj.writecm(cm)
        del fObj

    def partial_clear(self):
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
