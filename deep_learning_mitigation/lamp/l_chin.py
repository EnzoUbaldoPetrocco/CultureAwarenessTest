import sys
sys.path.insert(1, '../../')

from deep_learning_mitigation.classificator import ClassificatorClass
from deep_learning_mitigation.strings import Strings

strings = Strings()
paths = strings.lamp_paths
file_name = 'l_chin.csv'
for i in range(12,25):
    cc = ClassificatorClass(0, 0, paths,batch_size=4, fileName=file_name, verbose = 0, plot=0, validation_split=0.2, epochs=40, learning_rate=6e-5, lambda_index=i, times=10)
    cc.execute() 
