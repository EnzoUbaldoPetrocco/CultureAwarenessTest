import sys
sys.path.insert(1, '../../')

from standard.classificator import ClassificatorClass

paths = ['C:\\Users\\enzop\\Desktop\\FINALDS\\lamps\\chinese\\35\\Greyscale',
         'C:\\Users\\enzop\\Desktop\\FINALDS\\lamps\\french\\35\\Greyscale',
         'C:\\Users\\enzop\\Desktop\\FINALDS\\lamps\\turkish\\35\\Greyscale']

cc = ClassificatorClass(1,1,paths,'RFC',10)
cc.execute()