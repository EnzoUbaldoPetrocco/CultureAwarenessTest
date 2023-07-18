import sys
sys.path.insert(1, '../')
from numpy import abs
import numpy as np
import os
import pathlib
from Utils.utils import FileClass, ResultsClass
import glob
import colorama
from colorama import Fore

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

def get_err_std_for_every_lambda(accs_pu_cult):
    lerrs = [] # errors for each lambda
    lstds = [] # stds for each lambd
    try:
        for i in range(len(accs_pu_cult)):
            rrs , stds = [], []
            for j in range(3):
                acc, std = retrieve_mean_dev_std_accs_pu(accs_pu_cult[i][j])
                err = calc_ERR(acc/100)
                rrs.append(err)
                stds.append(std/100)
            lerrs.append(rrs)
            lstds.append(stds)
    except:
        ...
    return lerrs, lstds


def get_errs_stds_for_every_lambda(accs_pu):
    clerrs = [] # errors for each lambda for each culture
    clstds = [] # stds for each lambda for each cultures
    try:
        for i in range(3):
            lerrs, lstds = get_err_std_for_every_lambda(accs_pu[i])
            clerrs.append(lerrs)
            clstds.append(lstds)
    except:
        ...
    return clerrs, clstds

def get_lamb_for_min_CIC(errs_pu):
    clerrs = [] # errors for each lambda for each culture
    clstds = [] # stds for each lambda for each cultures
    for i in range(3):
        lerrs, lstds = get_err_std_for_every_lambda(errs_pu[i])
        clerrs.append(lerrs)
        clstds.append(lstds)
    CICs = []
    for i in range(len(errs_pu[0])):
        errors = []
        for j in range(3):
            errors.append(errs_pu[j][i])
        CIC = calc_CIC(errors)
        CICs.append(CIC)
    minCIC = min(CICs)
    lambda_index = CICs.index(minCIC)
    return lambda_index

def get_lamb_for_min_err(errs_pu, culture):
    errs_pu_c = errs_pu[culture]
    minimum = min(errs_pu_c)
    lambda_index = errs_pu_c.index(minimum)
    return lambda_index

def print_stats(errs_pu, stds_pu, lamb, accs):
    print(f'LAMBDA INDEX = {lamb}')
    errors = []
    for j in range(3):
        acc, _ = retrieve_mean_dev_std_accs_pu(accs[j][lamb])
        print(f'Accuracy is {acc:.4f}+-{stds_pu[j][lamb]:.4f} on Culture {j}')
        print(f'Error is {errs_pu[j][lamb]:.4f}+-{stds_pu[j][lamb]:.4f} on Culture {j}')
        errors.append(errs_pu[j][lamb])
    CIC = calc_CIC(errors)
    print(f'CIC is {CIC:.4f} on culture {j}\n\n')

def retrieve_statistics(p, model, ns, pu):
    print(Fore.WHITE + f'MODEL IS {model}')
    accs = retrievs_accs(p, model, ns, pu)
    clerrs, clstds = get_errs_stds_for_every_lambda(accs)
    for i in range(len(pu)):
        print(f'\nREFERRING TO PU={pu[i]}')
        # I want lambda for min errs (for each culture) and min CIC and their values
        accs_pu = accs[i]
        errs_pu, stds_pu = clerrs[i], clstds[i]
        lambdas = []
        if len(errs_pu)>0:
            for j in range(3):
                print(f'LAMBDA FOR MINIMUM ERROR ON CULTURE {j}')
                l = get_lamb_for_min_err(errs_pu, j)
                print_stats(errs_pu, stds_pu, l, accs_pu)
                lambdas.append(l)
            l = get_lamb_for_min_CIC(errs_pu)
            lambdas.append(l)
            print('LAMBDA FOR MINIMUM CIC')
            print_stats(errs_pu, stds_pu, l, accs_pu)
        



# TEST FUNCTIONS MITIGATION PART
# LAMPS
print(Fore.RED + '\n\nMITIGATION PART\n\n')
print(Fore.BLUE + 'LAMPS\n')
p = '../deep_learning_mitigation/lamp'
ns = 10
pu = ['0,05', '0,1']
# CHIN
model = 'l_chin'
retrieve_statistics(p, model, ns, pu)
# FREN
model = 'l_fren'
retrieve_statistics(p, model, ns, pu)
# TUR
model = 'l_tur'
retrieve_statistics(p, model, ns, pu)

print(Fore.BLUE + '\CARPETS STRETCHED\n')
p = '../deep_learning_mitigation/carpet_stretch'
ns = 10
pu = ['0,05', '0,1']
# CHIN
model = 'l_chin'
retrieve_statistics(p, model, ns, pu)
# FREN
model = 'l_fren'
retrieve_statistics(p, model, ns, pu)
# TUR
model = 'l_tur'
retrieve_statistics(p, model, ns, pu)

print(Fore.BLUE + '\CARPETS BLANKED\n')
p = '../deep_learning_mitigation/carpet_blanked'
ns = 10
pu = ['0,05', '0,1']
# CHIN
model = 'l_chin'
retrieve_statistics(p, model, ns, pu)
# FREN
model = 'l_fren'
retrieve_statistics(p, model, ns, pu)
# TUR
model = 'l_tur'
retrieve_statistics(p, model, ns, pu)





# TEST FUNCTIONS ANALYSIS PART
print(Fore.RED + '\n\n\nANALYSIS PART \n')
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
        print(f'Acc: {acc:.4f} on Culture {j}')
        er = calc_ERR(acc)
        errs.append(er)
        print(f'Error is {er:.4f}')
        print(f'With std:+-{std:.4f}')
    if len(errs)>=3:
        cic = calc_CIC(errs)
        print(f'CIC for this model is {cic:.4f}\n')

print(Fore.BLUE + '\nLAMPS\n')

# SVM
p = '../standard'
# pu = 0 and LIN
print(Fore.WHITE + '\nPU = 0')
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


print(Fore.BLUE + '\CARPETS STRETCHED\n')
print(Fore.WHITE + 'DL')
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


print(Fore.BLUE + '\CARPETS BLANKED\n')
print(Fore.WHITE + 'DL')
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