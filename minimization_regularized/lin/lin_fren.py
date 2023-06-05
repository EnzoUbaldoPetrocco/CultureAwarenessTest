import sys
sys.path.insert(1, '../../')

from minimization_regularized.classificator import ClassificatorClass
from standard.strings import Strings

strings = Strings()
paths = strings.paths

for i in range(25):
    cc = ClassificatorClass(1,1,paths,'SVC',30,'linear', fileName='lin_fren.csv', lambda_index=i)
    cc.execute()