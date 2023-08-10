import sys
sys.path.insert(1, '../../../')

from standard.classificator import ClassificatorClass
from standard.strings import Strings

strings = Strings()
paths = strings.paths

cc = ClassificatorClass(1,1,paths,'SVC',20,'linear', fileName='lin_fren.csv')
cc.execute_indipendent()