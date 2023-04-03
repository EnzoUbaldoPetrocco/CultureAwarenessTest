import sys
sys.path.insert(1, '../../')

from standard.classificator import ClassificatorClass

paths = ['C:\\Users\\enzop\\Desktop\\ARTICLE\\FINALDS\\lamps\\chinese\\35\\Greyscale',
         'C:\\Users\\enzop\\Desktop\\ARTICLE\\FINALDS\\lamps\\french\\35\\Greyscale',
         'C:\\Users\\enzop\\Desktop\\ARTICLE\\FINALDS\\lamps\\turkish\\35\\Greyscale']

cc = ClassificatorClass(0,1,paths,'SVC',30,'rbf', times = 2)
cc.execute_mixed([0,1])