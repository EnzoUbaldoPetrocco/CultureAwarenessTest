import sys
sys.path.insert(1, '../../')

from minimization_regularized.classificator import ClassificatorClass
from standard.strings import Strings

strings = Strings()
paths = strings.paths

for i in range(25):
    cc = ClassificatorClass(2,1,paths,'SVC',30,'linear', fileName='lin_tur.csv', lambda_index=i)
    cc.execute()