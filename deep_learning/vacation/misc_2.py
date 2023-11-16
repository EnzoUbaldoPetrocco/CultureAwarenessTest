import sys
sys.path.insert(1, '../../')

from deep_learning_mitigation.classificator import ClassificatorClass
from deep_learning_mitigation.strings import Strings
from time import sleep
from time import time

def main():

    strings = Strings()

    # CARPET
    print('STR Carpets mitigation')
    paths = strings.carpet_paths_str
    
    space = [0.05, 0.1]
    
    file_name = 'c_scan_mit.csv'
    culture = 2
    percent = space[1]
    cc = ClassificatorClass(culture,
                              0,
                              paths,
                              batch_size=2,
                              fileName=file_name,
                              verbose=0,
                              plot=0,
                              validation_split=0.2,
                              epochs=15,
                              gpu=True,
                              learning_rate=8e-5,
                              lambda_index=19,
                              times=1,
                              percent=percent)
    cc.execute()
    cc = None
    sleep(5)
    




      

if __name__ == "__main__":
    main()