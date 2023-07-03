import sys
sys.path.insert(1, '../../../')

from deep_learning.classificator import ClassificatorClass
from deep_learning.strings import Strings

strings = Strings()
paths = strings.carpet_paths_str
space = [0.1, 0.2, 0.5, 0.7, 0.8, 0.9]
for i in range(1,len(space)):
    j = space[i]
    cc = ClassificatorClass(2,0,paths,batch_size=4, fileName='c_scan.csv', verbose = 1, validation_split=0.2, epochs=10, learning_rate=4e-4, percent=j)
    cc.executenineone()