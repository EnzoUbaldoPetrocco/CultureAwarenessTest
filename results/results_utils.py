import sys
sys.path.insert(1, '../')
from numpy import abs
import numpy as np
import os
import pathlib
from Utils.utils import FileClass, ResultsClass
import glob

## CIC formula is:
# CIC = Ehat_C {|ERR^C - min_C'\inCit{ERR^C'}|}
# Ehat_C = 1/|Cit| * sum over C\inCit
def calc_CIC(ERRs):
    factor = 1/len(ERRs)
    CIC = 0
    for i in range(factor):
        CIC += abs(ERRs[i] - min(ERRs))
    CIC = factor*CIC
    return CIC

def calc_ERR(acc):
    return 1 - acc

def calc_std_dev(errors):
    return np.std(errors)

def is_out(path, model):
    l = path.split(model)
    n = l[1][0]
    out = l[1][1:len(l[1])]
    return n in out
    
def retrieve_accs_lamb(path, model, ns):
    paths = []
    accs = []
    for j in range(3):
        paths.extend(glob.glob(f'{path}/{model}{j}/'+'*.csv'))
    sorted(paths)
    rc = ResultsClass()
    for p in paths:
        if model in p:
            if is_out(p, model):
                fc = FileClass(p)
                accs_temp = fc.readcms()
                accs_temp = np.asarray(accs_temp[0:ns], dtype=object)  
                tot = rc.return_tot_elements(accs_temp[0])
                accs_temp = rc.calculate_percentage_confusion_matrix(accs_temp, tot)
                temp = []
                for i in range(ns):
                    temp.append(accs_temp[i][0][0]+accs_temp[i][1][1])
                accs.append(temp)
    return accs

def retrieve_accs_pu(path, model, ns):
    lamb_accs = []
    for lamb in range(-1,25):
        try:
            pth = path + f'/{lamb}'
            lamb_accs.append(retrieve_accs_lamb(pth, model, ns))
        except:
            print(f'Missing data for lamb index = {lamb}')
    return lamb_accs
    
def retrievs_accs(path, model, ns):
    pu = ['0,0','0,05', '0,1']
    pu_accs = []
    for p in pu:
        try:
            pth = path + f'/percent{p}'
            pu_accs.append(retrieve_accs_pu(pth, model, ns))
        except:
            print(f'Missing data for pu value = {p}')
    return pu_accs

def retrieve_mean_dev_std_accs_pu(l):
    m = np.mean(l)
    s = 0
    for i in l:
        s = s + (i-m)*(i-m)
    var = s / len(l)
    std = np.sqrt(var)
    return m, std
    
def sel_lamb(accs, pu_i, culture):
    accs = np.asarray(accs, dtype=object)
    accs_pu = np.asarray(accs[pu_i], dtype=object)
    means = []
    std_devs = []
    for lamb in range(len(accs_pu)):
        m, std = retrieve_mean_dev_std_accs_pu(accs_pu[lamb][culture])
        means.append(m)
        std_devs.append(std)
    max_acc = max(means)
    max_acc_ind = means.index(max_acc)
    return max_acc, max_acc_ind, std_devs[max_acc_ind]


# TEST FUNCTIONS MITIGATION PART
# LAMPS
print('LAMPS')
p = '../deep_learning_mitigation/lamp'
ns = 10
# CHIN
model = 'l_chin'
print(f'MODEL IS {model}')
accs = retrievs_accs(p, model, ns)
max_acc10, max_acc_ind10, std0 = sel_lamb(accs, 2, 0)
print(f'max acc for 10\% and its index for culture 0:')
print(f'Acc:{max_acc10}, index is:{max_acc_ind10}')
print(f'With error:+-{std0}')
max_acc10, max_acc_ind10, std0 = sel_lamb(accs, 2, 1)
print(f'max acc for 10\% and its index for culture 1:')
print(f'Acc:{max_acc10}, index is:{max_acc_ind10}')
print(f'With error:+-{std0}')
max_acc10, max_acc_ind10, std0 = sel_lamb(accs, 2, 2)
print(f'max acc for 10\% and its index for culture 2:')
print(f'Acc:{max_acc10}, index is:{max_acc_ind10}')
print(f'With error:+-{std0}')
# FREN
model = 'l_fren'
print(f'MODEL IS {model}')
accs = retrievs_accs(p, model, ns)
max_acc10, max_acc_ind10, std0 = sel_lamb(accs, 2, 0)
print(f'max acc for 10\% and its index for culture 0:')
print(f'Acc:{max_acc10}, index is:{max_acc_ind10}')
print(f'With error:+-{std0}')
max_acc10, max_acc_ind10, std0 = sel_lamb(accs, 2, 1)
print(f'max acc for 10\% and its index for culture 1:')
print(f'Acc:{max_acc10}, index is:{max_acc_ind10}')
print(f'With error:+-{std0}')
max_acc10, max_acc_ind10, std0 = sel_lamb(accs, 2, 2)
print(f'max acc for 10\% and its index for culture 2:')
print(f'Acc:{max_acc10}, index is:{max_acc_ind10}')
print(f'With error:+-{std0}')
# TUR
model = 'l_tur'
print(f'MODEL IS {model}')
accs = retrievs_accs(p, model, ns)
max_acc10, max_acc_ind10, std0 = sel_lamb(accs, 2, 0)
print(f'max acc for 10\% and its index for culture 0:')
print(f'Acc:{max_acc10}, index is:{max_acc_ind10}')
print(f'With error:+-{std0}')
max_acc10, max_acc_ind10, std0 = sel_lamb(accs, 2, 1)
print(f'max acc for 10\% and its index for culture 1:')
print(f'Acc:{max_acc10}, index is:{max_acc_ind10}')
print(f'With error:+-{std0}')
max_acc10, max_acc_ind10, std0 = sel_lamb(accs, 2, 2)
print(f'max acc for 10\% and its index for culture 2:')
print(f'Acc:{max_acc10}, index is:{max_acc_ind10}')
print(f'With error:+-{std0}')

