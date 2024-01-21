import sys

sys.path.insert(1, "../../../../")

from deep_learning_mitigation.ADVERSARY.complete.classificator import Midware
from deep_learning_mitigation.strings import Strings
import numpy as np
import gc
from time import sleep

culture = 0
strings = Strings()
paths = strings.carpet_paths_str
file_name = "c_ind.csv"
n = 5
percents = [0.05, 0.1]
lambda_indeces = range(6, 25)

mid = Midware(culture=culture,
                    greyscale=0,
                    paths=paths,
                    times=2,
                    fileName=file_name,
                    validation_split=0.2,
                    batch_size=2,
                    epochs=15,
                    learning_rate=4e-5,
                    verbose=0,
                    percent=percents[0],
                    plot = False,
                    run_eagerly = False,
                    lambda_index = 5,
                    gpu = True)
mid.execute(n)
mid = None
del mid
sleep(5)
gc.collect()


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
                    percent=percents[0],
                    plot = False,
                    run_eagerly = False,
                    lambda_index = lambda_index,
                    gpu = True)
    mid.execute(n)
    mid = None
    del mid
    sleep(5)
    gc.collect()

lambda_indeces = range(0, 25)
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
                    percent=percents[1],
                    plot = False,
                    run_eagerly = False,
                    lambda_index = lambda_index,
                    gpu = True)
    mid.execute(n)
    mid = None
    del mid
    sleep(5)
    gc.collect()
    

