import sys
sys.path.insert(1, '../../')

from deep_learning.classificator import ClassificatorClass
from deep_learning.strings import Strings

strings = Strings()
paths = strings.carpet_paths_bla

cc = ClassificatorClass(0, 0, paths,batch_size=2, fileName='c_ind_jap.csv',
                         verbose = 0, validation_split=0.3, epochs=8,
                           learning_rate=5e-4, gpu=False, times=25)
cc.execute_mixed([0,1])