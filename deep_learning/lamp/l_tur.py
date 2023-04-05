import sys
sys.path.insert(1, '../../')

from standard.classificator import ClassificatorClass
from deep_learning.strings import Strings

strings = Strings()
paths = strings.paths

cc = ClassificatorClass(2,1,paths, times = 30, verbose = 1, validation_split=0.2, epochs=10, learning_rate=4e-4)
cc.execute()