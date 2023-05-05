import sys
sys.path.insert(1, '/../')
from classificator import ClassificatorClass
import numpy as np

def extract_accuracies(base, root):
    cc = ClassificatorClass()
    for lambda_index in range(25):
        fileNames = []
        print(f'FOR LAMBDA INDEX EQUAL {lambda_index}')
        for l in range(3):
                    fileNamesOut = []
                    for o in range(3):
                        name = root + str(lambda_index) + '/' + base + str(
                            l) + f'/out{o}.csv'
                        
                        fileNamesOut.append(name)
                    fileNames.append(fileNamesOut)
        for i in range(3):
            for o in range(3):
                result = cc.get_results(fileNames[i][o])
                result = np.array(result, dtype=object)
                print(f'RESULTS ON CULTURE {i}, out {o}')
                tot = cc.resultsObj.return_tot_elements(result[0])
                pcm_list = cc.resultsObj.calculate_percentage_confusion_matrix(
                    result, tot)
                statistic = cc.resultsObj.return_statistics_pcm(pcm_list)
                accuracy = statistic[0][0][0] + statistic[0][1][1]
                print(f'Accuracy is {accuracy} %')

base = 'l_chin'
root = 'lamp/'
extract_accuracies(base, root)