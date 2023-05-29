import sys
sys.path.insert(1, '../../')

from deep_learning_mitigation.classificator import ClassificatorClass
from deep_learning_mitigation.strings import Strings

strings = Strings()
paths = strings.lamp_paths
file_name = 'l_fren.csv'
for i in range(-1,25):
    cc = ClassificatorClass(1,0,paths,batch_size=4, fileName=file_name, lambda_index=i, verbose = 0, plot=0, validation_split=0.2, epochs=40, learning_rate=6e-5, times=20)
    cc.execute()