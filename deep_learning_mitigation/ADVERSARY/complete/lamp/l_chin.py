import sys

sys.path.insert(1, "../../../../")

from deep_learning_mitigation.ADVERSARY.complete.classificator import Midware
from deep_learning_mitigation.strings import Strings
import numpy as np
import gc

culture = 0
strings = Strings()
paths = strings.lamp_paths
file_name = "l_chin.csv"
n = 5


mid = Midware(culture=culture,
                    greyscale=0,
                    paths=paths,
                    times=6,
                    fileName=file_name,
                    validation_split=0.2,
                    batch_size=2,
                    epochs=15,
                    learning_rate=4e-5,
                    verbose=0,
                    percent=0.1,
                    plot = False,
                    run_eagerly = False,
                    lambda_index = 0,
                    gpu = True)
mid.execute(n)
del mid
gc.collect()

percents = [0.05, 0.1]
lambda_indeces = range(0, 25)
for percent in percents:
    for lambda_index in lambda_indeces:
        mid = Midware(culture=culture,
                    greyscale=0,
                    paths=paths,
                    times=10,
                    fileName=file_name,
                    validation_split=0.2,
                    batch_size=2,
                    epochs=15,
                    learning_rate=4e-5,
                    verbose=0,
                    percent=percent,
                    plot = False,
                    run_eagerly = False,
                    lambda_index = lambda_index,
                    gpu = True)
        mid.execute(n)
        del mid
        gc.collect()
    

