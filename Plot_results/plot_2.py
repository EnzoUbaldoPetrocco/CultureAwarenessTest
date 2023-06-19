import sys

sys.path.insert(1, '../')
from deep_learning_mitigation.classificator import ClassificatorClass
import numpy as np
import os
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

def get_mean_accuracy(file_name):
    cc = ClassificatorClass(gpu=False)
    result = cc.get_results(file_name)
    result = np.array(result, dtype=object)
    tot = cc.resultsObj.return_tot_elements(result[0])
    pcm_list = cc.resultsObj.calculate_percentage_confusion_matrix(result, tot)
    statistic = cc.resultsObj.return_statistics_pcm(pcm_list)
    accuracy = statistic[0][0][0] + statistic[0][1][1]
    return accuracy

def get_l_folders(base):
    subs = []
    nums = []
    try: 
        for root, subdirs, files in os.walk(base):
            for sub in subdirs:
                d = os.path.basename(os.path.normpath(sub))
                subs.append(sub)
                num = int(d)
                nums.append(num)
    except:
        pass
    return subs, nums

def get_name_folders(base, name='percent'):
    subs = []
    nums = []
    try: 
        for root, subdirs, files in os.walk(base):
            d = os.path.basename(os.path.normpath(root))
            if name in d:
                subs.append(root)
                d = d.replace(name, '')
                d = d.replace(',', '.')
                num = float(d)
                nums.append(num)
    except:
        pass
    return subs, nums
        
def extract_model_mean_accuracies(model_path, model_folders, base):
    accs = []
    if model_folders is not None and len(model_folders)>0:
        for model_folder in model_folders:
            d = os.path.basename(os.path.normpath(base))
            m_fold = os.path.basename(os.path.normpath(model_folder))
            culture = m_fold.replace(base, '')
            culture = int(culture)
            for i in range(3):
                if culture == i:
                    out_file = model_folder + f'/out{i}.csv'
                    acc = get_mean_accuracy(out_file)
                    accs.append(acc)
    else:
        accs = [0,0,0]
    return accs
                
def compare_accs(act, max_ac):
    if act is not None and len(act)==3:
        for i in range(3):
            if act[i]>max_ac[i]:
                max_ac[i]=act[i]
    return max_ac

def main():
    ## TODO: 
# 1. scorrere lungo la lista di folder che iniziano per percent
# 2. per ogni file percent trovare tutte le sottocartelle
# 3. per ogni sotto cartella estrarre l'accuratezza dello specifico modello su ogni cultura
# 4. per ogni cultura estrarre i dati rispettivi alla cultura di riferimento
# 5. ciclare per ogni lambda (ogni sotto cartella del punto 2.) ed estrarre accuratezza massima
# 6. aggiungere quell'accuratezza alla lista delle percentuali
    chin_base = 'l_chin'
    fren_base = 'l_fren'
    tur_base = 'l_tur'
    root = f'../deep_learning_mitigation/lamp/'
    percents, nums = get_name_folders(root, 'percent')
    if percents is not None and len(percents)>0:
        for percent in percents:
            c_accs = []
            f_accs = []
            t_accs = []
            lambs, lnums = get_l_folders(percent + '/')
            chin_accs_max = [0,0,0]
            fren_accs_max = [0,0,0]
            tur_accs_max = [0,0,0]
            print(f'lambs are {lambs}')
            if lambs is not None and len(lambs)>0:
                for lam in lambs:
                    model_chin_res, model_chin_nums = get_name_folders(lam, name=chin_base)
                    model_fren_res, model_fren_nums = get_name_folders(lam, name=fren_base)
                    model_tur_res, model_tur_nums = get_name_folders(lam, name=tur_base)
                    #print(f'model_chin_res is {model_chin_res}')
                    #print(f'model_fren_res is {model_fren_res}')
                    #print(f'model_tur_res is {model_tur_res}')
                    chin_accs = extract_model_mean_accuracies(lam, model_chin_res, chin_base)
                    fren_accs = extract_model_mean_accuracies(lam, model_fren_res, fren_base)
                    tur_accs = extract_model_mean_accuracies(lam, model_tur_res, tur_base)
                    #print(f'chin_accs is {chin_accs}')
                    #print(f'fren_accs is {fren_accs}')
                    #print(f'tur_accs is {tur_accs}')
                    chin_accs_max = compare_accs(chin_accs,chin_accs_max)
                    fren_accs_max = compare_accs(fren_accs,fren_accs_max)
                    tur_accs_max = compare_accs(tur_accs,tur_accs_max)

            c_accs.append(chin_accs_max)
            f_accs.append(fren_accs_max)
            t_accs.append(tur_accs_max)
        print(c_accs)
        print(f_accs)
        print(t_accs)
                    
if __name__ == "__main__":
    main()
