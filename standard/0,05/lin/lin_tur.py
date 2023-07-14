import sys
sys.path.insert(1, '../../../')

from standard.classificator import ClassificatorClass
from standard.strings import Strings

strings = Strings()
paths = strings.paths

cc = ClassificatorClass(2,1,paths,'SVC',30,'linear', fileName='lin_tur.csv')
cc.executenineone(percent=0.05)