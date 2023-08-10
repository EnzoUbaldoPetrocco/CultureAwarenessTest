import sys
sys.path.insert(1, '../../../')

from deep_learning.classificator import ClassificatorClass
from deep_learning.strings import Strings

strings = Strings()
paths = strings.lamp_paths

cc = ClassificatorClass(1,0,paths, fileName='l_fren.csv', verbose = 1, validation_split=0.2, epochs=12, times=10)
cc.execute_indipendent()