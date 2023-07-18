import sys
sys.path.insert(1, '../../')

from deep_learning.classificator import ClassificatorClass
from deep_learning.strings import Strings
from time import sleep
import misc_2

strings = Strings()


# 5% of Minority Culture
percent = .05
# BLANKED CARPETS
paths = strings.carpet_paths_bla

# JAPANESE
cc = ClassificatorClass(1,0,paths,gpu=False,batch_size=4, fileName='c_jap_0,05bl.csv', verbose = 1, times=10, validation_split=0.2, epochs=15, learning_rate=5e-4, percent=percent)
cc.executenineone()
cc = None
sleep(5)

# SCANDINAVIAN
cc = ClassificatorClass(2,0,paths,gpu=False,batch_size=4, fileName='c_scan_0,05bl.csv', verbose = 1, times=10, validation_split=0.2, epochs=15, learning_rate=5e-4, percent=percent)
cc.executenineone()
cc = None
sleep(5)

# BLANKED CARPETS
paths = strings.carpet_paths_str
# INDIAN
cc = ClassificatorClass(0, 0, paths,gpu=False,batch_size=4, fileName='c_ind_0,05str.csv', verbose = 1, times=10, validation_split=0.2, epochs=15, learning_rate=5e-4, percent=percent)
cc.executenineone()
cc = None
sleep(5)

# JAPANESE
cc = ClassificatorClass(1,0,paths,gpu=False,batch_size=4, fileName='c_jap_0,05str.csv', verbose = 1, times=10, validation_split=0.2, epochs=15, learning_rate=5e-4, percent=percent)
cc.executenineone()
cc = None
sleep(5)

# SCANDINAVIAN
cc = ClassificatorClass(2,0,paths,gpu=False,batch_size=4, fileName='c_scan_0,05str.csv', verbose = 1, times=10, validation_split=0.2, epochs=15, learning_rate=5e-4, percent=percent)
cc.executenineone()
cc = None
sleep(5)

# LAMPS
paths = strings.lamp_paths

# CHINESE
cc = ClassificatorClass(0, 0, paths,batch_size=4, fileName='l_chin_0,05.csv', verbose = 1, times=10, validation_split=0.2, epochs=10, learning_rate=4e-4, percent=percent)
cc.executenineone()
cc = None
sleep(5)

# FRENCH
cc = ClassificatorClass(1,0,paths,batch_size=4, fileName='l_fren_0,05.csv', verbose = 1, times=10, validation_split=0.2, epochs=10, learning_rate=4e-4, percent=percent)
cc.executenineone()
cc = None
sleep(5)

# TURKISH
cc = ClassificatorClass(2,0,paths,batch_size=4, fileName='l_tur_0,05.csv', verbose = 1, times=10,validation_split=0.2, epochs=10, learning_rate=4e-4, percent=percent)
cc.executenineone()
cc = None
sleep(5)

# 10% of Minority Culture
percent=0.1
# CHINESE
cc = ClassificatorClass(0, 0, paths,batch_size=4, fileName='l_chin_0,1.csv', verbose = 1, times=10, validation_split=0.2, epochs=10, learning_rate=4e-4, percent=percent)
cc.executenineone()
cc = None
sleep(5)

# FRENCH
cc = ClassificatorClass(1,0,paths,batch_size=4, fileName='l_fren_0,1.csv', verbose = 1, times=10, validation_split=0.2, epochs=10, learning_rate=4e-4, percent=percent)
cc.executenineone()
cc = None
sleep(5)

# TURKISH
cc = ClassificatorClass(2,0,paths,batch_size=4, fileName='l_tur_0,1.csv', verbose = 1, times=10,validation_split=0.2, epochs=10, learning_rate=4e-4, percent=percent)
cc.executenineone()
cc = None
sleep(5)


# CHINESE
cc = ClassificatorClass(0, 0, paths,batch_size=4, fileName='l_chin.csv', verbose = 1, times=10, validation_split=0.2, epochs=10, learning_rate=4e-4, percent=percent)
cc.execute()
cc = None
sleep(5)

# FRENCH
cc = ClassificatorClass(1,0,paths,batch_size=4, fileName='l_fren.csv', verbose = 1, times=10, validation_split=0.2, epochs=10, learning_rate=4e-4, percent=percent)
cc.execute()
cc = None
sleep(5)

# TURKISH
cc = ClassificatorClass(2,0,paths,batch_size=4, fileName='l_tur.csv', verbose = 1, times=10,validation_split=0.2, epochs=10, learning_rate=4e-4, percent=percent)
cc.execute()
cc = None
sleep(5)

misc_2.main()