#!/usr/bin/env python
__author__ = "Enzo Ubaldo Petrocco"
import sys

sys.path.insert(1, "../")
from Processing.pipeline import Pipeline
import numpy as np


percents = [0.05, 0.1]

verbose_param = 1
proportions = [0.7, 0.2, 0.1]
cs = [0, 1, 2]
lamps = [0, 1]
oversamplings = [0, 1]
os_n = 100
adversarials = [1, 0]
gain = np.logspace(-4, 0, 5)
augments = [1, 0]
class_divisions = [0, 1]

for lamp in lamps:
    for c in cs:
        for pu in percents:
            for oversampling in oversamplings:
                for adversarial in adversarials:
                    for augment in augments:
                        if augment != adversarial:
                            #for g in gain:
                                g = gain[0]
                                if augment:
                                    pipe = Pipeline(
                                        search_root="../../../",
                                        dataset_path="../../../FINALDS/",
                                        lamp=lamp,
                                        save_root="./",
                                        verbose_param=verbose_param,
                                        shape=(100, 100, 3),
                                        n_cultures=3,
                                        majority_culture=c,
                                        pu=pu,
                                        proportions=proportions,
                                        oversampling=oversampling,
                                        os_n=os_n,
                                        adversarial=adversarial,
                                        epsilon=g,
                                        augment=augment,
                                        g=g,
                                    )
                                    pipe.plot_std_images()
                                if adversarial:
                                    for cls_div in class_divisions:
                                        pipe = Pipeline(
                                            search_root="../../../",
                                            dataset_path="../../../FINALDS/",
                                            lamp=lamp,
                                            save_root="./",
                                            verbose_param=verbose_param,
                                            shape=(100, 100, 3),
                                            n_cultures=3,
                                            majority_culture=c,
                                            pu=pu,
                                            proportions=proportions,
                                            oversampling=oversampling,
                                            os_n=os_n,
                                            adversarial=adversarial,
                                            epsilon=g,
                                            class_div=cls_div,
                                            augment=augment,
                                            g=g,
                                        )
                                        pipe.plot_adv_images()
