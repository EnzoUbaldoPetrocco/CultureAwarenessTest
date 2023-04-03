import sys
sys.path.insert(1, '../../')

from standard.classificator import ClassificatorClass

paths = ['C:\\Users\\enzop\\Desktop\\FINALDS\\lamps\\chinese\\35\\Greyscale',
         'C:\\Users\\enzop\\Desktop\\FINALDS\\lamps\\french\\35\\Greyscale',
         'C:\\Users\\enzop\\Desktop\\FINALDS\\lamps\\turkish\\35\\Greyscale']

cc = ClassificatorClass(2,1,paths,'SVC',10,'rbf')
cc.execute()