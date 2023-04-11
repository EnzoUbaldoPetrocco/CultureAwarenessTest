import sys
sys.path.insert(1, '../../')

from deep_learning_mitigation.classificator import ClassificatorClass
from deep_learning_mitigation.strings import Strings

strings = Strings()
paths = strings.carpet_paths_str

cc = ClassificatorClass(1,1,paths, fileName='c_jap.csv', verbose = 1, validation_split=0.2, epochs=10, learning_rate=4e-4)
cc.execute()