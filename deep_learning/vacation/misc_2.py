import sys
sys.path.insert(1, '../../')

from deep_learning.classificator import ClassificatorClass
from deep_learning.strings import Strings
from time import sleep

strings = Strings()





# 5% of Minority Culture
percent = .05
# BLANKED CARPETS
paths = strings.carpet_paths_bla
# INDIAN
cc = ClassificatorClass(0, 0, paths,batch_size=4, fileName='c_ind_0,05bl.csv', verbose = 1, times=10, validation_split=0.2, epochs=15, learning_rate=5e-4, percent=percent)
cc.executenineone()
cc = None
sleep(5)

# JAPANESE
cc = ClassificatorClass(1,0,paths,batch_size=4, fileName='c_jap_0,1bl.csv', verbose = 1, times=10, validation_split=0.2, epochs=15, learning_rate=5e-4, percent=percent)
cc.executenineone()
cc = None
sleep(5)

# SCANDINAVIAN
cc = ClassificatorClass(2,0,paths,batch_size=4, fileName='c_scan_0,1bl.csv', verbose = 1, times=10, validation_split=0.2, epochs=15, learning_rate=5e-4, percent=percent)
cc.executenineone()
cc = None
sleep(5)

# BLANKED CARPETS
paths = strings.carpet_paths_str
# INDIAN
cc = ClassificatorClass(0, 0, paths,batch_size=4, fileName='c_ind_0,1str.csv', verbose = 1, times=10, validation_split=0.2, epochs=15, learning_rate=5e-4, percent=percent)
cc.executenineone()
cc = None
sleep(5)

# JAPANESE
cc = ClassificatorClass(1,0,paths,batch_size=4, fileName='c_jap_0,1str.csv', verbose = 1, times=10, validation_split=0.2, epochs=15, learning_rate=5e-4, percent=percent)
cc.executenineone()
cc = None
sleep(5)

# SCANDINAVIAN
cc = ClassificatorClass(2,0,paths,batch_size=4, fileName='c_scan_0,1str.csv', verbose = 1, times=10, validation_split=0.2, epochs=15, learning_rate=5e-4, percent=percent)
cc.executenineone()
cc = None
sleep(5)

# LAMPS
paths = strings.lamp_paths

# CHINESE
cc = ClassificatorClass(0, 0, paths,batch_size=4, fileName='l_chin_0,2.csv', verbose = 1, times=10, validation_split=0.2, epochs=10, learning_rate=4e-4, percent=percent)
cc.executenineone()
cc = None
sleep(5)

# FRENCH
cc = ClassificatorClass(1,0,paths,batch_size=4, fileName='l_fren_0,2.csv', verbose = 1, times=10, validation_split=0.2, epochs=10, learning_rate=4e-4, percent=percent)
cc.executenineone()
cc = None
sleep(5)

# TURKISH
cc = ClassificatorClass(2,0,paths,batch_size=4, fileName='l_tur_0,2.csv', verbose = 1, times=10,validation_split=0.2, epochs=10, learning_rate=4e-4, percent=percent)
cc.executenineone()
cc = None
sleep(5)

# CHINESE
cc = ClassificatorClass(0, 0, paths,batch_size=4, fileName='l_chin_0,2.csv', verbose = 1, times=10, validation_split=0.2, epochs=10, learning_rate=4e-4, percent=percent)
cc.executenineone()
cc = None
sleep(5)

# FRENCH
cc = ClassificatorClass(1,0,paths,batch_size=4, fileName='l_fren_0,2.csv', verbose = 1, times=10, validation_split=0.2, epochs=10, learning_rate=4e-4, percent=percent)
cc.executenineone()
cc = None
sleep(5)

# TURKISH
cc = ClassificatorClass(2,0,paths,batch_size=4, fileName='l_tur_0,2.csv', verbose = 1, times=10,validation_split=0.2, epochs=10, learning_rate=4e-4, percent=percent)
cc.executenineone()
cc = None
sleep(5)

# CHINESE
cc = ClassificatorClass(0, 0, paths,batch_size=4, fileName='l_chin_0,2.csv', verbose = 1, times=10, validation_split=0.2, epochs=10, learning_rate=4e-4, percent=percent)
cc.executenineone()
cc = None
sleep(5)

# FRENCH
cc = ClassificatorClass(1,0,paths,batch_size=4, fileName='l_fren_0,2.csv', verbose = 1, times=10, validation_split=0.2, epochs=10, learning_rate=4e-4, percent=percent)
cc.executenineone()
cc = None
sleep(5)

# TURKISH
cc = ClassificatorClass(2,0,paths,batch_size=4, fileName='l_tur_0,2.csv', verbose = 1, times=10,validation_split=0.2, epochs=10, learning_rate=4e-4, percent=percent)
cc.executenineone()
cc = None
sleep(5)