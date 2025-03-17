#!/usr/bin/env python
__author__ = "Enzo Ubaldo Petrocco"
import sys

sys.path.insert(1, "../")
from Processing.pipeline import Pipeline
import numpy as np
import tensorflow as tf




bc = tf.keras.losses.BinaryCrossentropy(
    from_logits=True,
    label_smoothing=0.0,
    axis=-1,
    reduction='sum_over_batch_size',
    name='binary_crossentropy'
)

acc = tf.keras.metrics.BinaryAccuracy(
    name='binary_accuracy', dtype=None, threshold=0.5
)

percents = [0.2]

verbose_param = 1
proportions = [0.7, 0.2, 0.1]
cs = [0, 1, 2]
lamps = [0, 1]
oversamplings = [0]
os_n = 150
adversarials = [1, 0]
gain = np.logspace(-4, 0, 5)
augments = [1, 0]
class_divisions = [1, 0]

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
                                    ls, vs, ts = pipe.preprocessing()
                                    pipe.model_selection(ls, vs, 2, bc, acc)
                                    for culture in range(pipe.n_cultures):
                                        ts[c][1] = np.asarray(ts[c][1])[pipe.n_cultures]
                                        res = pipe.error_estimation(ts[c])
                                        pt_to_append = f"culture_{c}/"
                                    pipe.save_results(res, False, pt_to_append)
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
                                        ls, vs, ts = pipe.preprocessing()
                                        pipe.model_selection(ls, vs, 2, bc, acc)
                                        for culture in range(pipe.n_cultures):
                                            ts[c][1] = np.asarray(ts[c][1])[pipe.n_cultures]
                                            res = pipe.error_estimation(ts[c])
                                            pt_to_append = f"culture_{c}/"
                                        pipe.save_results(res, False, pt_to_append)
