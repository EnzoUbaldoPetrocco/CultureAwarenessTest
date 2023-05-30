import sys
sys.path.insert(1, '../')
from deep_learning_mitigation.classificator import ClassificatorClass
import numpy as np
import os



def get_right_folders(base):
    root, subdirs, files = os.walk(base)
    subs = []
    lambdas = []
    for dir in subdirs:
        try:
            d = os.path.basename(os.path.normpath(dir))
            d = int(d)
            subs.append(dir)
            lambdas.append(d)
        except:
            pass
    return subs, lambdas

def get_mean_accuracy(file_name):
    cc = ClassificatorClass()
    result = cc.get_results(file_name)
    result = np.array(result, dtype=object)
    tot = cc.resultsObj.return_tot_elements(result[0])
    pcm_list = cc.resultsObj.calculate_percentage_confusion_matrix(
        result, tot)
    statistic = cc.resultsObj.return_statistics_pcm(pcm_list)
    accuracy = statistic[0][0][0] + statistic[0][1][1]
    return accuracy

def extract_accuracies(base, root):
    subfolders, lambdas = get_right_folders(root)
    # For each lambda I have to extract accuracy for each culture
    # and for each output
    accuracies = []
    for j,sub in enumerate(subfolders):
        fileNames = []
        print(f'FOR LAMBDA INDEX EQUAL {lambdas[j]}')
        for l in range(3):
                    fileNamesOut = []
                    for o in range(3):
                        name = root + str(lambdas[j]) + '/' + base + str(
                            l) + f'/out{o}.csv'
                        
                        fileNamesOut.append(name)
                    fileNames.append(fileNamesOut)
        accuracies_culture = []
        for i in range(3):
            accuracies_output = []
            for o in range(3):
                acc = get_mean_accuracy(fileNames[i][o])
                accuracies_output.append(acc)
            accuracies_culture.append(accuracies_output)
        accuracies.append(accuracies_culture)
    
def main():
    base = 'l_chin'
    root = '../deep_learning_mitigation/lamp/'
    accs = extract_accuracies(base, root)
    print(accs)

if __name__ == "__main__":
    main()


