import sys
sys.path.insert(1, '../')
from deep_learning_mitigation.classificator import ClassificatorClass
import numpy as np
import os
from matplotlib import pyplot as plt



def get_right_folders(base):
    subs = []
    lambdas = []
    for root, subdirs, files in os.walk(base):
        #print(f'Root is {root}')
        #print(f'Subdirs is {subdirs}')
        #print(f'files is {files}')
        try:
            d = os.path.basename(os.path.normpath(root))
            d = int(d)
            subs.append(root)
            lambdas.append(d)
        except:
            pass
    return subs, lambdas

def get_mean_accuracy(file_name):
    cc = ClassificatorClass(gpu=False)
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
    for j in range(len(lambdas)):
        fileNames = []
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
                print(fileNames[i][o])
                acc = get_mean_accuracy(fileNames[i][o])
                accuracies_output.append(acc)
            accuracies_culture.append(accuracies_output)
        accuracies.append(accuracies_culture)
    return accuracies

def extract_accuracies_on_culture(base, root, culture):
    temp_accs = extract_accuracies(base,root)
    accs = []
    for acc in temp_accs:
        a = acc[culture][culture]
        accs.append(a)
    return accs

def plot_acc(title, accs, refs): 
    fig, ax = plt.subplots()
    plt.xlabel("Lambda parameter")
    plt.ylabel("Accuracy")
    miny = min(accs+refs)
    maxy = max(accs+refs)
    plt.ylim([miny-0.5,maxy+0.5])
    plt.grid(True)
    n_points = len(accs)
    ax.set_title(title)
    ticks_label = []
    logspac= np.logspace(-3,1,25)
    ticks_label.append(0)
    for i in range(0,n_points-1):
        ticks_label.append(str(logspac[i])[0:6])
    x = np.linspace(0, 100, n_points)
    plt.xticks(ticks=x, labels=ticks_label)
    ax.plot(x, accs, linewidth=2.0)
    for ref in refs:
        plt.axhline(y=ref, color='r', linestyle='-')
    #ax.set_xscale('log')
    plt.show()

def plot_accuracies_per_culture(base, root, ref_c1, ref_c2, ref_c3, pre_title):
    chin_accs = extract_accuracies_on_culture(base, root, 0)
    #print(chin_accs)
    fren_accs = extract_accuracies_on_culture(base, root, 1)
    #print(fren_accs)
    tur_accs = extract_accuracies_on_culture(base, root, 2)
    #print(tur_accs)
    plot_acc(f'{pre_title} Accuracy on Chinese Culture', chin_accs, ref_c1)
    plot_acc(f'{pre_title} Accuracy on French Culture', fren_accs, ref_c2)
    plot_acc(f'{pre_title} Accuracy on Turkish Culture', tur_accs, ref_c3)

def extract_accuracies_per_test_culture(base,root, culture):
    temp_accs = extract_accuracies(base,root)
    accs = []
    for acc in temp_accs:
        a = acc[culture]
        accs.append(a)
    return accs

def plot_all_accuracies(base, root):
    chin_on_chin = extract_accuracies_per_test_culture(base, root, 0)
    fren_on_fren = extract_accuracies_per_test_culture(base, root, 1)
    turn_on_tur = extract_accuracies_per_test_culture(base, root, 2)
    chin_on_chin = np.asarray(chin_on_chin, dtype=object)
    fren_on_fren = np.asarray(fren_on_fren, dtype=object)
    turn_on_tur = np.asarray(turn_on_tur, dtype=object)
    plot_acc('Accuracy on Chinese Culture out 0', chin_on_chin[:,0])
    plot_acc('Accuracy on Chinese Culture out 1', chin_on_chin[:,1])
    plot_acc('Accuracy on Chinese Culture out 2', chin_on_chin[:,2])
    plot_acc('Accuracy on French Culture out 0', fren_on_fren[:,0])
    plot_acc('Accuracy on French Culture out 1', fren_on_fren[:,1])
    plot_acc('Accuracy on French Culture out 2', fren_on_fren[:,2])
    plot_acc('Accuracy on Turkish Culture out 0', turn_on_tur[:,0])
    plot_acc('Accuracy on Turkish Culture out 1', turn_on_tur[:,1])
    plot_acc('Accuracy on Turkish Culture out 2', turn_on_tur[:,2])
    
def main():
    chin_base = 'l_chin'
    fren_base = 'l_fren'
    tur_base = 'l_tur'
    root = f'../deep_learning_mitigation/lamp/'
    chin_on_chin_ref = 87.21
    chin_on_fren_ref = 83.79
    chin_on_tur_ref = 75.33
    fren_on_chin_ref = 81.04
    fren_on_fren_ref = 89.22
    fren_on_tur_ref = 74.60
    tur_on_chin_ref = 78.14
    tur_on_fren_ref = 79.33
    tur_on_tur_ref = 89.13
    plot_accuracies_per_culture(chin_base, root, [chin_on_chin_ref],[chin_on_fren_ref,fren_on_fren_ref],[chin_on_tur_ref, tur_on_tur_ref], 'Chinese Model' )
    plot_accuracies_per_culture(fren_base, root, [fren_on_fren_ref,fren_on_chin_ref],[fren_on_fren_ref],[ fren_on_tur_ref,tur_on_tur_ref], 'French Model' )
    plot_accuracies_per_culture(tur_base, root, [chin_on_chin_ref,tur_on_chin_ref],[tur_on_fren_ref,fren_on_fren_ref],[tur_on_tur_ref], 'Turkish Model' )
    #plot_all_accuracies(base, root)
    


if __name__ == "__main__":
    main()

