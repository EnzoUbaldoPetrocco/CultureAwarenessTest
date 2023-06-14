import sys
sys.path.insert(1, '../../')

from deep_learning_mitigation.classificator import ClassificatorClass
from deep_learning_mitigation.strings import Strings

strings = Strings()
paths = strings.lamp_paths
file_name = 'l_tur.csv'
culture = 2
space = [0.1, 0.2, 0.5, 0.7, 0.8, 0.9]
for i in range(-1, 25):
    for j in range(3, 6):
        percent = space[j]
        cc = ClassificatorClass(culture,
                                0,
                                paths,
                                batch_size=4,
                                fileName=file_name,
                                verbose=0,
                                plot=0,
                                validation_split=0.2,
                                epochs=40,
                                learning_rate=6e-5,
                                lambda_index=i,
                                times=20,
                                percent=percent)
        cc.execute()