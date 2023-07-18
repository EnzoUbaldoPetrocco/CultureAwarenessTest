import sys
sys.path.insert(1, '../../')

from deep_learning_mitigation.classificator import ClassificatorClass
from deep_learning_mitigation.strings import Strings
from time import sleep

def main():
    # LAMPS
    print('Lamps mitigation')
    strings = Strings()
    paths = strings.lamp_paths
    file_name = 'l_chin_mit.csv'
    culture = 0
    space = [0.05]
    for i in range(-1, 25):
      percent = space[0]
      cc = ClassificatorClass(culture,
                              0,
                              paths,
                              batch_size=4,
                              fileName=file_name,
                              verbose=0,
                              plot=0,
                              validation_split=0.2,
                              epochs=30,
                              learning_rate=6e-5,
                              lambda_index=i,
                              times=10,
                              percent=percent)
      cc.execute()
      cc = None
      sleep(5)
    file_name = 'l_fren_mit.csv'
    culture = 1
    for i in range(-1, 25):
      percent = space[0]
      cc = ClassificatorClass(culture,
                              0,
                              paths,
                              batch_size=4,
                              fileName=file_name,
                              verbose=0,
                              plot=0,
                              validation_split=0.2,
                              epochs=30,
                              learning_rate=6e-5,
                              lambda_index=i,
                              times=10,
                              percent=percent)
      cc.execute()
      cc = None
      sleep(5)
    file_name = 'l_tur_mit.csv'
    culture = 2
    for i in range(-1, 25):
      percent = space[0]
      cc = ClassificatorClass(culture,
                              0,
                              paths,
                              batch_size=4,
                              fileName=file_name,
                              verbose=0,
                              plot=0,
                              validation_split=0.2,
                              epochs=30,
                              learning_rate=6e-5,
                              lambda_index=i,
                              times=10,
                              percent=percent)
      cc.execute()
      cc = None
      sleep(5)

    # CARPET
    print('STR Carpets mitigation')
    paths = strings.carpet_paths_str
    file_name = 'c_ind_mit.csv'
    culture = 0
    space = [0.05, 0.1]
    for i in range(-1, 25):
      percent = space[0]
      cc = ClassificatorClass(culture,
                              0,
                              paths,
                              batch_size=4,
                              gpu=False,
                              fileName=file_name,
                              verbose=0,
                              plot=0,
                              validation_split=0.2,
                              epochs=30,
                              learning_rate=6e-5,
                              lambda_index=i,
                              times=10,
                              percent=percent)
      cc.execute()
      cc = None
      sleep(5)
      percent = space[1]
      cc = ClassificatorClass(culture,
                              0,
                              paths,
                              batch_size=4,
                              fileName=file_name,
                              verbose=0,
                              gpu=False,
                              plot=0,
                              validation_split=0.2,
                              epochs=30,
                              learning_rate=6e-5,
                              lambda_index=i,
                              times=10,
                              percent=percent)
      cc.execute()
      cc = None
      sleep(5)
    file_name = 'c_jap_mit.csv'
    culture = 1
    for i in range(-1, 25):
      percent = space[0]
      cc = ClassificatorClass(culture,
                              0,
                              paths,
                              batch_size=4,
                              fileName=file_name,
                              verbose=0,
                              gpu=False,
                              plot=0,
                              validation_split=0.2,
                              epochs=30,
                              learning_rate=6e-5,
                              lambda_index=i,
                              times=10,
                              percent=percent)
      cc.execute()
      cc = None
      sleep(5)
      percent = space[1]
      cc = ClassificatorClass(culture,
                              0,
                              paths,
                              batch_size=4,
                              fileName=file_name,
                              verbose=0,
                              gpu=False,
                              plot=0,
                              validation_split=0.2,
                              epochs=30,
                              learning_rate=6e-5,
                              lambda_index=i,
                              times=10,
                              percent=percent)
      cc.execute()
      cc = None
      sleep(5)
    file_name = 'c_scan_mit.csv'
    culture = 2
    for i in range(-1, 25):
      percent = space[0]
      cc = ClassificatorClass(culture,
                              0,
                              paths,
                              batch_size=4,
                              fileName=file_name,
                              verbose=0,
                              plot=0,
                              validation_split=0.2,
                              epochs=30,
                              gpu=False,
                              learning_rate=6e-5,
                              lambda_index=i,
                              times=10,
                              percent=percent)
      cc.execute()
      cc = None
      sleep(5)
      percent = space[1]
      cc = ClassificatorClass(culture,
                              0,
                              paths,
                              batch_size=4,
                              fileName=file_name,
                              verbose=0,
                              plot=0,
                              validation_split=0.2,
                              epochs=30,
                              gpu=False,
                              learning_rate=6e-5,
                              lambda_index=i,
                              times=10,
                              percent=percent)
      cc.execute()
      cc = None
      sleep(5)

    # CARPET
    print('BLA Carpets mitigation')
    paths = strings.carpet_paths_bla
    file_name = 'c_ind_mit.csv'
    culture = 0
    space = [0.05, 0.1]
    for i in range(-1, 25):
      percent = space[0]
      cc = ClassificatorClass(culture,
                              0,
                              paths,
                              batch_size=4,
                              gpu=False,
                              fileName=file_name,
                              verbose=0,
                              plot=0,
                              validation_split=0.2,
                              epochs=30,
                              learning_rate=6e-5,
                              lambda_index=i,
                              times=10,
                              percent=percent)
      cc.execute()
      cc = None
      sleep(5)
      percent = space[1]
      cc = ClassificatorClass(culture,
                              0,
                              paths,
                              batch_size=4,
                              fileName=file_name,
                              verbose=0,
                              gpu=False,
                              plot=0,
                              validation_split=0.2,
                              epochs=30,
                              learning_rate=6e-5,
                              lambda_index=i,
                              times=10,
                              percent=percent)
      cc.execute()
      cc = None
      sleep(5)
    file_name = 'c_jap_mit.csv'
    culture = 1
    for i in range(-1, 25):
      percent = space[0]
      cc = ClassificatorClass(culture,
                              0,
                              paths,
                              batch_size=4,
                              fileName=file_name,
                              verbose=0,
                              gpu=False,
                              plot=0,
                              validation_split=0.2,
                              epochs=30,
                              learning_rate=6e-5,
                              lambda_index=i,
                              times=10,
                              percent=percent)
      cc.execute()
      cc = None
      sleep(5)
      percent = space[1]
      cc = ClassificatorClass(culture,
                              0,
                              paths,
                              batch_size=4,
                              fileName=file_name,
                              verbose=0,
                              gpu=False,
                              plot=0,
                              validation_split=0.2,
                              epochs=30,
                              learning_rate=6e-5,
                              lambda_index=i,
                              times=10,
                              percent=percent)
      cc.execute()
      cc = None
      sleep(5)
    file_name = 'c_scan_mit.csv'
    culture = 2
    for i in range(-1, 25):
      percent = space[0]
      cc = ClassificatorClass(culture,
                              0,
                              paths,
                              batch_size=4,
                              fileName=file_name,
                              verbose=0,
                              plot=0,
                              validation_split=0.2,
                              epochs=30,
                              gpu=False,
                              learning_rate=6e-5,
                              lambda_index=i,
                              times=10,
                              percent=percent)
      cc.execute()
      cc = None
      sleep(5)
      percent = space[1]
      cc = ClassificatorClass(culture,
                              0,
                              paths,
                              batch_size=4,
                              fileName=file_name,
                              verbose=0,
                              plot=0,
                              validation_split=0.2,
                              epochs=30,
                              gpu=False,
                              learning_rate=6e-5,
                              lambda_index=i,
                              times=10,
                              percent=percent)
      cc.execute()
      cc = None
      sleep(5)
      

if __name__ == "__main__":
    main()