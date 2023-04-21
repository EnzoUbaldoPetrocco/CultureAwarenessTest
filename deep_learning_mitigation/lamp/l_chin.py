import sys
sys.path.insert(1, '../../')

from deep_learning_mitigation.classificator import ClassificatorClass
from deep_learning_mitigation.strings import Strings

strings = Strings()
paths = strings.lamp_paths

cc = ClassificatorClass(0, 0, paths,batch_size=8, fileName='l_chin.csv', verbose = 1, validation_split=0.15, epochs=10, learning_rate=2e-5, plot= True)
cc.execute()