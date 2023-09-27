import sys
sys.path.insert(1, '../../')

from deep_learning.classificator import ClassificatorClass
from deep_learning.strings import Strings
from time import sleep
import misc_2

strings = Strings()


# 5% of Minority Culture
percent = .05
paths = strings.carpet_paths_str

# JAPANESE
cc = ClassificatorClass(1,0,paths,batch_size=4, fileName='c_jap_0,05str.csv', verbose = 0, times=8, validation_split=0.2, epochs=15, learning_rate=5e-4, percent=percent)
cc.executenineone_model_selection()
cc = None
sleep(5)

# SCANDINAVIAN
cc = ClassificatorClass(2,0,paths,batch_size=4, fileName='c_scan_0,05str.csv', verbose = 0, times=10, validation_split=0.2, epochs=15, learning_rate=5e-4, percent=percent)
cc.executenineone_model_selection()
cc = None
sleep(5)

# 10% of Minority Culture
percent = .1
paths = strings.carpet_paths_str
# INDIAN
cc = ClassificatorClass(0, 0, paths,batch_size=4, fileName='c_ind_0,1str.csv', verbose = 0, times=10, validation_split=0.2, epochs=15, learning_rate=5e-4, percent=percent)
cc.executenineone_model_selection()
cc = None
sleep(5)

# JAPANESE
cc = ClassificatorClass(1,0,paths,batch_size=4, fileName='c_jap_0,1str.csv', verbose = 0, times=10, validation_split=0.2, epochs=15, learning_rate=5e-4, percent=percent)
cc.executenineone_model_selection()
cc = None
sleep(5)

# SCANDINAVIAN
cc = ClassificatorClass(2,0,paths,batch_size=4, fileName='c_scan_0,1str.csv', verbose = 0, times=10, validation_split=0.2, epochs=15, learning_rate=5e-4, percent=percent)
cc.executenineone_model_selection()
cc = None
sleep(5)

#misc_2.main()