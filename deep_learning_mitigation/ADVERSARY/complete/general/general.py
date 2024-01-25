import sys

sys.path.insert(1, "../../../../")

from deep_learning_mitigation.ADVERSARY.complete.classificator import Midware
from deep_learning_mitigation.strings import Strings
import numpy as np
import gc
from time import sleep


strings = Strings()

def main():
    culture = int(sys.argv[1])
    if sys.argv[2]=="carpets":
        paths = strings.carpet_paths_str
    if sys.argv[2]=="lamps":
        paths = strings.lamp_paths
    file_name = str(sys.argv[3])
    times = int(sys.argv[4])
    percent = float(sys.argv[5])
    lambda_index = int(sys.argv[6])
    mid = Midware(culture=culture,
                        greyscale=0,
                        paths=paths,
                        times=times,
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
    mid.execute(5)

if __name__=="__main__":
    main()

    

