import sys

sys.path.insert(1, "../../../../")

from deep_learning_mitigation.ADVERSARY.complete.classificator import Midware
from deep_learning_mitigation.strings import Strings
import numpy as np

culture = 0
strings = Strings()
paths = strings.carpet_paths_str
file_name = "c_ind.csv"
n = 5
percents = [0.05, 0.1]
lambda_indeces = [[11, 18, 18],[14, 14, 14]]
for percent in percents:
    for lambda_index in lambda_indeces:
        mid = Midware(culture=culture,
                        greyscale=0,
                        paths=paths,
                        times=1,
                        fileName=file_name,
                        validation_split=0.2,
                        batch_size=4,
                        epochs=15,
                        learning_rate=4e-5,
                        verbose=0,
                        percent=percent,
                        plot = False,
                        run_eagerly = False,
                        lambda_index = 0,
                        gpu = True)
        mid.execute(n)
    

