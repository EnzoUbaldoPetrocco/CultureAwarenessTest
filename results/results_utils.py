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
    for i in range(len(ERRs)):
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
    
def retrievs_accs(path, model, ns, pu):
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
        try:
            l = accs_pu[lamb]
            l = l[culture]
            m, std = retrieve_mean_dev_std_accs_pu(l)
            means.append(m)
            std_devs.append(std)
        except:
            ...
            #print('not enough information in one culture')
    if means:
        max_acc = max(means)
        max_acc_ind = means.index(max_acc)
        return max_acc, max_acc_ind, std_devs[max_acc_ind]
    return None, None, None

def print_errors_CIC_mitigation(p, model, ns, pu):
    print(f'MODEL IS {model}')
    accs = retrievs_accs(p, model, ns, pu)
    for i in range(len(pu)):
        print(f'\nREFERRING TO PU={pu[i]}')
        errs = []
        for j in range(3):
            max_acc10, max_acc_ind10, std0 = sel_lamb(accs, i, j)
            if max_acc10:
                print(f'max acc for 10\% and its index for culture {j}:')
                print(f'Acc: {max_acc10/100}, index is:{max_acc_ind10/100}')
                er = calc_ERR(max_acc10/100)
                errs.append(er)
                print(f'Error is {er}')
                print(f'With std:+-{std0/100}')
        if len(errs)>=3:
            cic = calc_CIC(errs)
            print(f'CIC for this model is {cic}\n')



# TEST FUNCTIONS MITIGATION PART
# LAMPS
print('MITIGATION PART')
print('\nLAMPS\n')
p = '../deep_learning_mitigation/lamp'
ns = 10
pu = ['0,05', '0,1']
# CHIN
model = 'l_chin'
print_errors_CIC_mitigation(p, model, ns, pu)
# FREN
model = 'l_fren'
print_errors_CIC_mitigation(p, model, ns, pu)
# TUR
model = 'l_tur'
print_errors_CIC_mitigation(p, model, ns, pu)

print('\CARPETS STRETCHED\n')
p = '../deep_learning_mitigation/carpet_stretch'
ns = 10
pu = ['0,05', '0,1']
# CHIN
model = 'l_chin'
print_errors_CIC_mitigation(p, model, ns, pu)
# FREN
model = 'l_fren'
print_errors_CIC_mitigation(p, model, ns, pu)
# TUR
model = 'l_tur'
print_errors_CIC_mitigation(p, model, ns, pu)

print('\CARPETS BLANKED\n')
p = '../deep_learning_mitigation/carpet_blanked'
ns = 10
pu = ['0,05', '0,1']
# CHIN
model = 'l_chin'
print_errors_CIC_mitigation(p, model, ns, pu)
# FREN
model = 'l_fren'
print_errors_CIC_mitigation(p, model, ns, pu)
# TUR
model = 'l_tur'
print_errors_CIC_mitigation(p, model, ns, pu)


# TEST FUNCTIONS ANALYSIS PART
print('\n\nANALYSIS PART \n\n')
def retrieve_accs_standard(path, model, ns):
    paths = []
    accs = []
    for j in range(3):
        paths.extend(glob.glob(f'{path}/{model}{j}.csv'))
    sorted(paths)
    rc = ResultsClass()
    for p in paths:
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

def print_errors_CIC(p, model, ns):
    print(f'MODEL IS {model}')
    accs = retrieve_accs_standard(p, model, ns)
    errs = []
    for j in range(3):
        l = np.multiply(accs[j], 0.01)  
        acc, std = retrieve_mean_dev_std_accs_pu(l)
        print(f'Acc: {acc} on Culture {j}')
        er = calc_ERR(acc)
        errs.append(er)
        print(f'Error is {er}')
        print(f'With std:+-{std}')
    if len(errs)>=3:
        cic = calc_CIC(errs)
        print(f'CIC for this model is {cic}\n')

print('\nLAMPS\n')

# SVM
p = '../standard'
# pu = 0 and LIN
print('\nPU = 0')
print('LSVM')
pt = p + '/lin'
model = 'lin_chin'
print_errors_CIC(pt, model, ns)
model = 'lin_fren'
print_errors_CIC(pt, model, ns)
model = 'lin_tur'
print_errors_CIC(pt, model, ns)

# pu = 0 and RBF
print('GSVM')
pt = p + '/rbf'
model = 'rbf_chin'
print_errors_CIC(pt, model, ns)
model = 'rbf_fren'
print_errors_CIC(pt, model, ns)
model = 'rbf_tur'
print_errors_CIC(pt, model, ns)

# pu = 0 and DL
print('DL')
pt = '../deep_learning' + '/lamp'
model = 'l_chin'
print_errors_CIC(pt, model, ns)
model = 'l_fren'
print_errors_CIC(pt, model, ns)
model = 'l_tur'
print_errors_CIC(pt, model, ns)


print('\nPU = 0.1')
# pu = 0.1 and LIN
print('LSVM')
pt = p + '/9010' + '/lin'
model = 'lin_chin'
print_errors_CIC(pt, model, ns)
model = 'lin_fren'
print_errors_CIC(pt, model, ns)
model = 'lin_tur'
print_errors_CIC(pt, model, ns)

# pu = 0.1 and RBF
print('GSVM')
pt = p + '/9010' + '/rbf'
model = 'rbf_chin'
print_errors_CIC(pt, model, ns)
model = 'rbf_fren'
print_errors_CIC(pt, model, ns)
model = 'rbf_tur'
print_errors_CIC(pt, model, ns)

# pu = 0.1 and DL
print('DL')
pt = '../deep_learning' + '/9010/lamp/percent0,1'
model = 'l_chin'
print_errors_CIC(pt, model, ns)
model = 'l_fren'
print_errors_CIC(pt, model, ns)
model = 'l_tur'
print_errors_CIC(pt, model, ns)

print('\nPU = 0.05')
# pu = 0.05 and LIN
print('LSVM')
pt = p + '/0,05' + '/lin'
model = 'lin_chin'
print_errors_CIC(pt, model, ns)
model = 'lin_fren'
print_errors_CIC(pt, model, ns)
model = 'lin_tur'
print_errors_CIC(pt, model, ns)

# pu = 0.1 and RBF
print('GSVM')
pt = p + '/0,05' + '/rbf'
model = 'rbf_chin'
print_errors_CIC(pt, model, ns)
model = 'rbf_fren'
print_errors_CIC(pt, model, ns)
model = 'rbf_tur'
print_errors_CIC(pt, model, ns)

# pu = 0.05 and DL
print('DL')
pt = '../deep_learning' + '/9010/lamp/percent0,05'
model = 'l_chin'
print_errors_CIC(pt, model, ns)
model = 'l_fren'
print_errors_CIC(pt, model, ns)
model = 'l_tur'
print_errors_CIC(pt, model, ns)


print('\CARPETS STRETCHED\n')
print('DL')
print('\nPU = 0.0')
pt = '../deep_learning' + '/carpet_stretch'
model = 'l_chin'
print_errors_CIC(pt, model, ns)
model = 'l_fren'
print_errors_CIC(pt, model, ns)
model = 'l_tur'
print_errors_CIC(pt, model, ns)
print('\nPU = 0.1')
pt = '../deep_learning' + '/9010/carpet_stretch/percent0,1'
model = 'l_chin'
print_errors_CIC(pt, model, ns)
model = 'l_fren'
print_errors_CIC(pt, model, ns)
model = 'l_tur'
print_errors_CIC(pt, model, ns)
print('\nPU = 0.05')
pt = '../deep_learning' + '/9010/carpet_stretch/percent0,05'
model = 'l_chin'
print_errors_CIC(pt, model, ns)
model = 'l_fren'
print_errors_CIC(pt, model, ns)
model = 'l_tur'
print_errors_CIC(pt, model, ns)


print('\CARPETS BLANKED\n')
print('DL')
print('\nPU = 0.0')
pt = '../deep_learning' + '/carpet_blanked'
model = 'l_chin'
print_errors_CIC(pt, model, ns)
model = 'l_fren'
print_errors_CIC(pt, model, ns)
model = 'l_tur'
print_errors_CIC(pt, model, ns)
print('\nPU = 0.1')
pt = '../deep_learning' + '/9010/carpet_blanked/percent0,1'
model = 'l_chin'
print_errors_CIC(pt, model, ns)
model = 'l_fren'
print_errors_CIC(pt, model, ns)
model = 'l_tur'
print_errors_CIC(pt, model, ns)
print('\nPU = 0.05')
pt = '../deep_learning' + '/9010/carpet_blanked/percent0,05'
model = 'l_chin'
print_errors_CIC(pt, model, ns)
model = 'l_fren'
print_errors_CIC(pt, model, ns)
model = 'l_tur'
print_errors_CIC(pt, model, ns)