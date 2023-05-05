import sys
sys.path.insert(1, '../../')

from deep_learning_mitigation.classificator import ClassificatorClass
from deep_learning_mitigation.strings import Strings

strings = Strings()
paths = strings.lamp_paths
file_name = 'l_chin.csv'
for i in range(25):
    cc = ClassificatorClass(0, 0, paths,batch_size=4, fileName=file_name, verbose = 1, validation_split=0.2, epochs=1, learning_rate=4e-4, times=2, lambda_index=i)
    cc.execute()
