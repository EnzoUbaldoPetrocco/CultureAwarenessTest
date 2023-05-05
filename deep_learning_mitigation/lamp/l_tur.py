import sys
sys.path.insert(1, '../../')

from deep_learning_mitigation.classificator import ClassificatorClass
from deep_learning_mitigation.strings import Strings

strings = Strings()
paths = strings.lamp_paths
file_name = 'l_tur.csv'
for i in range(25):
    cc = ClassificatorClass(2,0,paths,batch_size=4, fileName=file_name, lambda_index=i, verbose = 1, validation_split=0.2, epochs=20, learning_rate=2e-5)
    cc.execute()