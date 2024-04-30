#!/usr/bin/env python
__author__ = "Enzo Ubaldo Petrocco"
import sys

sys.path.insert(1, "../../")

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from Utils.Results.Results import ResAcquisitionClass


class VisualizerClass:
    def plot_table(self, df, title):
        """
        This function plots a DataFrame as Table
        :param df: DataFrame
        :param title: title of Table
        """
        fig, ax = plt.subplots()
        # hide axes
        fig.patch.set_visible(False)
        ax.axis("off")
        ax.axis("tight")
        tab = ax.table(cellText=df.values, colLabels=df.columns, loc="center")
        tab.auto_set_font_size(False)
        tab.set_fontsize(9.5)
        fig.tight_layout()
        ax.set_title(title)
        plt.show()


def print_tables(standards, lamps, cultures, percents, augments, adversary, lambda_indeces, taugments, tadversaries, test_g_augs, test_eps, paths, visobj):
    for standard in standards:
        for lamp in lamps:
            for culture in cultures:
                for percent in range(len(percents)):
                    for augment in augments:
                        for adv in adversary:
                            if standard:
                                for taugment in taugments:
                                    for tadversary in tadversaries:
                                        if taugment and tadversary:
                                            for tgaug in range(len(test_g_augs)):
                                                for teps in range(len(test_eps)):

                                                    pt = paths[standard][lamp][culture][
                                                        percent
                                                    ][augment][adv][taugment][tadversary][
                                                        tgaug
                                                    ][
                                                        teps
                                                    ]

                                                    pt = pt + "res.csv"
                                                    df = pd.read_csv(pt)
                                                    visobj.plot_table(
                                                        df[df.columns[1 : len(df.columns)]],
                                                        pt,
                                                    )
                                        if taugment and not tadversary:
                                            for tgaug in range(len(test_g_augs)):
                                                pt = paths[standard][lamp][culture][
                                                        percent
                                                    ][augment][adv][taugment][tadversary][
                                                        tgaug
                                                    ][
                                                        teps
                                                    ]
                                                pt = pt + "res.csv"
                                                df = pd.read_csv(pt)
                                                visobj.plot_table(
                                                    df[df.columns[1 : len(df.columns)]],
                                                    pt,
                                                )

                                        if not taugment and tadversary:
                                            for teps in range(len(test_eps)):
                                                pt = paths[standard][lamp][culture][
                                                        percent
                                                    ][augment][adv][taugment][tadversary][
                                                        tgaug
                                                    ][
                                                        teps
                                                    ]
                                                pt = pt + "res.csv"
                                                df = pd.read_csv(pt)
                                                visobj.plot_table(
                                                    df[df.columns[1 : len(df.columns)]],
                                                    pt,
                                                )
                                        if not taugment and not tadversary:
                                            pt = paths[standard][lamp][culture][
                                                    percent
                                                ][augment][adv][taugment][tadversary][
                                                    tgaug
                                                ][
                                                    teps
                                                ]
                                            pt = pt + "res.csv"
                                            df = pd.read_csv(pt)
                                            visobj.plot_table(
                                                df[df.columns[1 : len(df.columns)]],
                                                pt,
                                            )
                            else:
                                for lambda_index in lambda_indeces:
                                    for taugment in taugments:
                                        for tadversary in tadversaries:
                                            if taugment and tadversary:
                                                for tgaug in range(len(test_g_augs)):
                                                    for teps in range(len(test_eps)):
                                                        pt = paths[standard][lamp][culture][
                                                                percent
                                                            ][augment][adv][lambda_index][
                                                                taugment
                                                            ][
                                                                tadversary
                                                            ][
                                                                tgaug
                                                            ][
                                                                teps
                                                            ]

                                                        pt = pt + "res.csv"
                                                        df = pd.read_csv(pt)
                                                        visobj.plot_table(
                                                            df[df.columns[1 : len(df.columns)]],
                                                            pt,
                                                        )
                                            if taugment and not tadversary:
                                                for tgaug in range(len(test_g_augs)):
                                                    pt = paths[standard][lamp][culture][
                                                        percent
                                                    ][augment][adv][lambda_index][
                                                        taugment
                                                    ][
                                                        tadversary
                                                    ][
                                                        tgaug
                                                    ][
                                                        teps
                                                    ]
                                                    pt = pt + "res.csv"
                                                    df = pd.read_csv(pt)
                                                    visobj.plot_table(
                                                        df[df.columns[1 : len(df.columns)]],
                                                        pt,
                                                    )
                                            if not taugment and tadversary:
                                                for teps in range(len(test_eps)):
                                                    pt = paths[standard][lamp][culture][
                                                        percent
                                                    ][augment][adv][lambda_index][
                                                        taugment
                                                    ][
                                                        tadversary
                                                    ][
                                                        tgaug
                                                    ][
                                                        teps
                                                    ]
                                                    pt = pt + "res.csv"
                                                    df = pd.read_csv(pt)
                                                    visobj.plot_table(
                                                        df[df.columns[1 : len(df.columns)]],
                                                        pt,
                                                    )
                                            if not taugment and not tadversary:
                                                tgaug = 0
                                                teps = 0
                                                pt = paths[standard][lamp][culture][
                                                        percent
                                                    ][augment][adv][lambda_index][
                                                        taugment
                                                    ][
                                                        tadversary
                                                    ][
                                                        tgaug
                                                    ][
                                                        teps
                                                    ]
                                                pt = pt + "res.csv"
                                                df = pd.read_csv(pt)
                                                visobj.plot_table(
                                                    df[df.columns[1 : len(df.columns)]],
                                                    pt,
                                                )

def from_perstr2float(val):
    val = val.split("%")
    val = val[0]
    val = float(val)
    return val

def plot(stddf, mitdfs, title='', save=False, path='./'):
    """
    This function plots in CIC-ERR plane the CIC(ERR) value of standard model
    and the CIC(ERR, gamma) values of mitigated model
    :param stddf: DataFrame of standard model
    :param mitdfs: DataFrame of mitigated model
    :param save: if enable, saves the image in path
    :param path: path in which the image is saved
    """

    X = []
    y = []
    X_err = []
    y_err = []
    for i in range(len(mitdfs)):
        X.append(from_perstr2float(mitdfs[i]['ERR'][0]))
        y.append(from_perstr2float(mitdfs[i]['CIC'][0]))
        X_err.append(from_perstr2float(mitdfs[i]['ERR std'][0]))
        y_err.append(from_perstr2float(mitdfs[i]['CIC std'][0]))
        #print(mitdfs[i]['ERR'][0])
        #print(mitdfs[i]['CIC'][0])
        #print(mitdfs[i]['ERR std'][0])
        #print(mitdfs[i]['CIC std'][0])
        #print("\n\n")

    
    # Plotting both the curves simultaneously 
    plt.scatter(from_perstr2float(stddf['ERR'][0]), from_perstr2float(stddf['CIC'][0]), color='r', label='Standard Model') 
    plt.plot(X, y, color='k', label='Mitigated Model') 

    # Plot errors
    plt.errorbar(from_perstr2float(stddf['ERR'][0]), from_perstr2float(stddf['CIC'][0]), xerr=from_perstr2float(stddf['ERR std'][0]), yerr=from_perstr2float(stddf['CIC std'][0]), fmt="ro")
    plt.errorbar(X, y, xerr=X_err, yerr=y_err, fmt="ko")
    
    # Naming the x-axis, y-axis and the whole graph 
    plt.xlabel("ERR") 
    plt.ylabel("CIC") 
    plt.title(title) 

    plt.xlim((0, 75))
    plt.ylim((0, 30))
    
    # Adding legend, which helps us recognize the curve according to it's color 
    plt.legend() 
    
    # To load the display window 
    plt.show() 
    


def gen_title_plot(lamp, culture, percent, augment, adversary, taugment, tadversary, gaug, geps):
    test_g_augs = [0.01, 0.05, 0.1]
    test_eps = [0.0005, 0.001, 0.005]
    if lamp:
        if culture==0:
            title = "Chinese Model"
        if culture==1:
            title = "French Model"
        if culture==2:
            title = "Turkish Model"
    else:
        if culture==0:
            title = "Indian Model"
        if culture==1:
            title = "Japanese Model"
        if culture==2:
            title = "Scandinavian Model"

    title += " with μ" + f"={percent}\n"
    if augment:
        if adversary:
            title += "Using TOTAUG in training "
        else:
            title += "Using STDAUG in training "
    else:
        if adversary:
            title += "Using ADVAUG in training "
        else:
            title += "Using NOAUG in training "

    if taugment:
        if tadversary:
            title += f"Tested on TOTAUG with GAUG={gaug} and ε={geps}"  
        else:
            title += f"Tested on STDAUG with GAUG={gaug}"  
    else:
        if tadversary:
            title += f"Tested on ADVAUG with ε={geps}"  
        else:
            title += f"Tested on NOAUG"  

    return title

def graphic_lambdas_comparison(standards, lamps, cultures, percents, augments, adversary, lambda_indeces, taugments, tadversaries, test_g_augs, test_eps, resacqobj:ResAcquisitionClass, basePath='../Results/'):
        for lamp in lamps:
            for culture in cultures:
                for percent in percents:
                    for augment in augments:
                        for adv in adversary:
                            for taugment in taugments:
                                for tadversary in tadversaries:
                                        tgaug = None
                                        teps = None
                                        if taugment and tadversary:
                                            for tgaug in test_g_augs:
                                                for teps in test_eps:
                                                    pt = resacqobj.buildPath(
                                                            basePath,
                                                            1,
                                                            'DL',
                                                            lamp,
                                                            culture,
                                                            percent,
                                                            augment,
                                                            adv,
                                                            0,
                                                            taugment,
                                                            tadversary,
                                                            tgaug,
                                                            teps,
                                                        )
                                                    
                                                    pt = pt.split('/')
                                                    pt = pt[0:len(pt)-2]
                                                    stdpt = ""
                                                    for p in pt:
                                                        stdpt += p + "/"
                                                    stdpt += "res.csv"
                                                    stddf = pd.read_csv(stdpt)
                                                    mitdfs = []
                                                    for lambda_index in lambda_indeces:
                                                        pt = resacqobj.buildPath(
                                                            basePath,
                                                            0,
                                                            'DL',
                                                            lamp,
                                                            culture,
                                                            percent,
                                                            augment,
                                                            adv,
                                                            lambda_index,
                                                            taugment,
                                                            tadversary,
                                                            tgaug,
                                                            teps,
                                                        )
                                                        pt = pt.split('/')
                                                        pt = pt[:len(pt)-2]
                                                        mitpt = ''
                                                        for p in pt:
                                                            mitpt += p + "/"
                                                        pt = mitpt + "res.csv"
                                                        df = pd.read_csv(pt)
                                                        mitdfs.append(df)

                                                    title = gen_title_plot(lamp, culture, percent, augment, adversary, taugment, tadversary, tgaug, teps)
                                                    plot(stddf=stddf, mitdfs=mitdfs, title=title)

                                        if taugment and not tadversary:
                                            for tgaug in test_g_augs:
                                                pt = resacqobj.buildPath(
                                                            basePath,
                                                            1,
                                                            'DL',
                                                            lamp,
                                                            culture,
                                                            percent,
                                                            augment,
                                                            adv,
                                                            0,
                                                            taugment,
                                                            tadversary,
                                                            tgaug,
                                                            teps,
                                                        )
                                                    
                                                pt = pt.split('/')
                                                pt = pt[0:len(pt)-2]
                                                stdpt = ""
                                                for p in pt:
                                                    stdpt += p + "/"
                                                stdpt += "res.csv"
                                                stddf = pd.read_csv(stdpt)
                                                mitdfs = []
                                                for lambda_index in lambda_indeces:
                                                        pt = resacqobj.buildPath(
                                                            basePath,
                                                            0,
                                                            'DL',
                                                            lamp,
                                                            culture,
                                                            percent,
                                                            augment,
                                                            adv,
                                                            lambda_index,
                                                            taugment,
                                                            tadversary,
                                                            tgaug,
                                                            teps,
                                                        )
                                                        pt = pt.split('/')
                                                        pt = pt[:len(pt)-2]
                                                        mitpt = ''
                                                        for p in pt:
                                                            mitpt += p + "/"
                                                        pt = mitpt + "res.csv"
                                                        df = pd.read_csv(pt)
                                                        mitdfs.append(df)

                                                title = gen_title_plot(lamp, culture, percent, augment, adversary, taugment, tadversary, tgaug, teps)
                                                plot(stddf=stddf, mitdfs=mitdfs, title=title)

                                        if not taugment and tadversary:
                                            for teps in test_eps:
                                                pt = resacqobj.buildPath(
                                                            basePath,
                                                            1,
                                                            'DL',
                                                            lamp,
                                                            culture,
                                                            percent,
                                                            augment,
                                                            adv,
                                                            0,
                                                            taugment,
                                                            tadversary,
                                                            tgaug,
                                                            teps,
                                                        )
                                                pt = pt.split('/')
                                                pt = pt[0:len(pt)-2]
                                                stdpt = ""
                                                for p in pt:
                                                    stdpt += p + "/"
                                                stdpt += "res.csv"
                                                stddf = pd.read_csv(stdpt)
                                                mitdfs = []
                                                for lambda_index in lambda_indeces:
                                                        pt = resacqobj.buildPath(
                                                            basePath,
                                                            0,
                                                            'DL',
                                                            lamp,
                                                            culture,
                                                            percent,
                                                            augment,
                                                            adv,
                                                            lambda_index,
                                                            taugment,
                                                            tadversary,
                                                            tgaug,
                                                            teps,
                                                        )
                                                        pt = pt.split('/')
                                                        pt = pt[:len(pt)-2]
                                                        mitpt = ''
                                                        for p in pt:
                                                            mitpt += p + "/"
                                                        pt = mitpt + "res.csv"
                                                        df = pd.read_csv(pt)
                                                        mitdfs.append(df)

                                                title = gen_title_plot(lamp, culture, percent, augment, adversary, taugment, tadversary, tgaug, teps)
                                                plot(stddf=stddf, mitdfs=mitdfs, title=title)

                                        if (not taugment) and (not tadversary):
                                            pt = resacqobj.buildPath(
                                                            basePath,
                                                            1,
                                                            'DL',
                                                            lamp,
                                                            culture,
                                                            percent,
                                                            augment,
                                                            adv,
                                                            0,
                                                            taugment,
                                                            tadversary,
                                                            tgaug,
                                                            teps,
                                                        )
                                            pt = pt.split('/')
                                            pt = pt[0:len(pt)-2]
                                            stdpt = ""
                                            for p in pt:
                                                stdpt += p + "/"
                                            stdpt += "res.csv"
                                            stddf = pd.read_csv(stdpt)
                                            mitdfs = []
                                            
                                            for lambda_index in lambda_indeces:
                                                        pt = resacqobj.buildPath(
                                                            basePath,
                                                            0,
                                                            'DL',
                                                            lamp,
                                                            culture,
                                                            percent,
                                                            augment,
                                                            adv,
                                                            lambda_index,
                                                            taugment,
                                                            tadversary,
                                                            tgaug,
                                                            teps,
                                                        )
                                                        pt = pt.split('/')
                                                        pt = pt[:len(pt)-2]
                                                        mitpt = ''
                                                        for p in pt:
                                                            mitpt += p + "/"
                                                        pt = mitpt + "res.csv"
                                                        df = pd.read_csv(pt)
                                                        mitdfs.append(df)
                                            title = gen_title_plot(lamp, culture, percent, augment, adversary, taugment, tadversary, tgaug, teps)
                                            plot(stddf=stddf, mitdfs=mitdfs, title=title)

                                        
def main():
    visobj = VisualizerClass()
    resacqobj = ResAcquisitionClass()


    standards = [0, 1]
    alg = "DL"
    lamps = [0, 1]
    cultures = [0, 1, 2]
    percents = [0.05, 0.1]
    augments = [0, 1]
    adversary = [0, 1]
    lambda_indeces = range(-1, 13)
    taugments = [0, 1]
    tadversaries = [0, 1]
    test_g_augs = [0.01, 0.05, 0.1]
    test_eps = [0.0005, 0.001, 0.005]
    t_cults = [0, 1, 2]

    #print_tables(standards, lamps, cultures, percents, augments, adversary, lambda_indeces, taugments, tadversaries, test_g_augs, test_eps, paths, visobj)
    graphic_lambdas_comparison(standards, lamps, cultures, percents, augments, adversary, lambda_indeces, taugments, tadversaries, test_g_augs, test_eps, resacqobj)
    


if __name__ == "__main__":
    main()
