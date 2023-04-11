import sys
sys.path.insert(1, '../../')

from deep_learning.classificator import ClassificatorClass
from deep_learning.strings import Strings

strings = Strings()
paths = strings.lamp_paths

cc = ClassificatorClass(0, 0, paths, fileName='l_chin_fren.csv', verbose = 1, validation_split=0.25, epochs=10, learning_rate=4e-4, plot= True)
cc.execute_mixed([0,1])