import sys
sys.path.insert(1, '../../../')

from deep_learning.classificator import ClassificatorClass
from deep_learning.strings import Strings

strings = Strings()
paths = strings.lamp_paths
space = [0.0, 0.05, 0.1]
for i in range(0,len(space)):
    j = space[i]
    cc = ClassificatorClass(1,0,paths,batch_size=4, fileName='c_jap.csv', verbose = 0, validation_split=0.3, epochs=7, learning_rate=4e-4, percent=j)
    cc.executenineone_model_selection()