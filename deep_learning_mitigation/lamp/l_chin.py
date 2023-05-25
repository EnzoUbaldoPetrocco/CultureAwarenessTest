import sys
sys.path.insert(1, '../../')

from deep_learning_mitigation.classificator import ClassificatorClass
from deep_learning_mitigation.strings import Strings

strings = Strings()
paths = strings.lamp_paths
file_name = 'l_chin.csv'
for i in range(11,25):
    cc = ClassificatorClass(0, 0, paths,batch_size=4, fileName=file_name, verbose = 0, validation_split=0.2, epochs=20, learning_rate=2e-5, lambda_index=i)
    cc.execute()
