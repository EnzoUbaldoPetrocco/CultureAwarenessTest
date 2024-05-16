#!/usr/bin/env python
__author__ = "Enzo Ubaldo Petrocco"
import sys


sys.path.insert(1, "../../")

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from Utils.Results.Results import ResAcquisitionClass
from Utils.FileManager.FileManager import FileManagerClass


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


def print_tables(
    standards,
    lamps,
    cultures,
    percents,
    augments,
    adversary,
    lambda_indeces,
    taugments,
    tadversaries,
    test_g_augs,
    test_eps,
    paths,
    visobj,
):
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
                                                    ][augment][adv][taugment][
                                                        tadversary
                                                    ][
                                                        tgaug
                                                    ][
                                                        teps
                                                    ]

                                                    pt = pt + "res.csv"
                                                    df = pd.read_csv(pt)
                                                    visobj.plot_table(
                                                        df[
                                                            df.columns[
                                                                1 : len(df.columns)
                                                            ]
                                                        ],
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
                                                        pt = paths[standard][lamp][
                                                            culture
                                                        ][percent][augment][adv][
                                                            lambda_index
                                                        ][
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
                                                            df[
                                                                df.columns[
                                                                    1 : len(df.columns)
                                                                ]
                                                            ],
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
                                                        df[
                                                            df.columns[
                                                                1 : len(df.columns)
                                                            ]
                                                        ],
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
                                                        df[
                                                            df.columns[
                                                                1 : len(df.columns)
                                                            ]
                                                        ],
                                                        pt,
                                                    )
                                            if not taugment and not tadversary:
                                                tgaug = 0
                                                teps = 0
                                                pt = paths[standard][lamp][culture][
                                                    percent
                                                ][augment][adv][lambda_index][taugment][
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


def perstr2float(val):
    val = val.split("%")
    val = val[0]
    val = float(val)
    return val


def plotandsave(stddf, mitdfs, plot=True, title="", save=False, path="./"):
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
        X.append(perstr2float(mitdfs[i]["ERR"][0]))
        y.append(perstr2float(mitdfs[i]["CIC"][0]))
        X_err.append(perstr2float(mitdfs[i]["ERR std"][0]))
        y_err.append(perstr2float(mitdfs[i]["CIC std"][0]))

    # Plotting both the curves simultaneously
    plt.scatter(
        perstr2float(stddf["ERR"][0]),
        perstr2float(stddf["CIC"][0]),
        color="r",
        label="Standard Model",
    )
    plt.plot(X, y, color="k", label="Mitigated Model")

    # Plot errors
    plt.errorbar(
        perstr2float(stddf["ERR"][0]),
        perstr2float(stddf["CIC"][0]),
        xerr=perstr2float(stddf["ERR std"][0]),
        yerr=perstr2float(stddf["CIC std"][0]),
        fmt="ro",
    )
    plt.errorbar(X, y, xerr=X_err, yerr=y_err, fmt="ko")

    # Naming the x-axis, y-axis and the whole graph
    plt.xlabel("ERR")
    plt.ylabel("CIC")
    plt.title(title)

    plt.xlim((0, 100))
    plt.ylim((0, 30))

    # Adding legend, which helps us recognize the curve according to it's color
    plt.legend()

    # To load the display window
    if plot:
        plt.show()

    if save:
        plt.savefig(path)

    # print(path)
    plt.close()


def gen_title_plot(
    lamp, culture, percent, augment, adversary, taugment, tadversary, gaug, geps
):
    test_g_augs = [0.01, 0.05, 0.1]
    test_eps = [0.0005, 0.001, 0.005]
    svpath = "./"
    if lamp:
        if culture == 0:
            title = "Chinese Model"
            svpath = "LC/"
        if culture == 1:
            title = "French Model"
            svpath = "LF/"
        if culture == 2:
            title = "Turkish Model"
            svpath = "LT/"
    else:
        if culture == 0:
            title = "Indian Model"
            svpath = "CI/"
        if culture == 1:
            title = "Japanese Model"
            svpath = "CJ/"
        if culture == 2:
            title = "Scandinavian Model"
            svpath = "CS/"

    title += " with μ" + f"={percent}\n"
    svpath += f"{percent}/"
    if augment:
        if adversary:
            title += "Using TOTAUG in training "
            svpath += "TOTAUG/"
        else:
            title += "Using STDAUG in training "
            svpath += "STDAUG/"
    else:

        if adversary:
            title += "Using ADVAUG in training "
            svpath += "ADVAUG/"
        else:
            title += "Using NOAUG in training "
            svpath += "NOAUG/"

    if taugment:
        if tadversary:
            title += f"Tested on TOTAUG with GAUG={gaug} and ε={geps}"
            svpath += f"TTOTAUG/GAUG={gaug}/EPS={geps}/"
        else:
            title += f"Tested on STDAUG with GAUG={gaug}"
            svpath += f"TSTDAUG/GAUG={gaug}/"
    else:
        if tadversary:
            title += f"Tested on ADVAUG with ε={geps}"
            svpath += f"TADVAUG/EPS={geps}/"
        else:
            title += f"Tested on NOAUG"
            svpath += f"TNOAUG/"

    fObj = FileManagerClass(svpath)
    del fObj
    svpath += "img.png"
    # (f"svpath is {svpath}")
    return title, svpath


def graphic_lambdas_comparison(
    standards,
    lamps,
    cultures,
    percents,
    augments,
    adversary,
    lambda_indeces,
    taugments,
    tadversaries,
    test_g_augs,
    test_eps,
    resacqobj: ResAcquisitionClass,
    basePath="../Results/",
    plot=False,
):
    for lamp in lamps:
        for culture in cultures:
            for percent in percents:
                for augment in augments:
                    for adv in adversary:
                        for taugment in taugments:
                            for tadversary in tadversaries:
                                if taugment and tadversary:
                                    for tgaug in test_g_augs:
                                        for teps in test_eps:
                                            pt = resacqobj.buildPath(
                                                basePath=basePath,
                                                standard=1,
                                                alg="DL",
                                                lamp=lamp,
                                                culture=culture,
                                                percent=percent,
                                                augment=augment,
                                                adversary=adv,
                                                lambda_index=0,
                                                taugment=taugment,
                                                tadversary=tadversary,
                                                tgaug=tgaug,
                                                teps=teps,
                                            )

                                            pt = pt.split("/")
                                            pt = pt[0 : len(pt) - 2]
                                            stdpt = ""
                                            for p in pt:
                                                stdpt += p + "/"
                                            stdpt += "res.csv"
                                            stddf = pd.read_csv(stdpt)
                                            mitdfs = []
                                            for lambda_index in lambda_indeces:
                                                pt = resacqobj.buildPath(
                                                    basePath=basePath,
                                                    standard=0,
                                                    alg="DL",
                                                    lamp=lamp,
                                                    culture=culture,
                                                    percent=percent,
                                                    augment=augment,
                                                    adversary=adv,
                                                    lambda_index=lambda_index,
                                                    taugment=taugment,
                                                    tadversary=tadversary,
                                                    tgaug=tgaug,
                                                    teps=teps,
                                                )
                                                pt = pt.split("/")
                                                pt = pt[: len(pt) - 2]
                                                mitpt = ""
                                                for p in pt:
                                                    mitpt += p + "/"
                                                pt = mitpt + "res.csv"
                                                df = pd.read_csv(pt)
                                                mitdfs.append(df)

                                            title, svpath = gen_title_plot(
                                                lamp,
                                                culture,
                                                percent,
                                                augment,
                                                adv,
                                                taugment,
                                                tadversary,
                                                tgaug,
                                                teps,
                                            )
                                            plotandsave(
                                                stddf=stddf,
                                                mitdfs=mitdfs,
                                                plot=plot,
                                                title=title,
                                                save=True,
                                                path=svpath,
                                            )

                                if taugment and not tadversary:
                                    for tgaug in test_g_augs:
                                        pt = resacqobj.buildPath(
                                            basePath=basePath,
                                            standard=1,
                                            alg="DL",
                                            lamp=lamp,
                                            culture=culture,
                                            percent=percent,
                                            augment=augment,
                                            adversary=adv,
                                            lambda_index=0,
                                            taugment=taugment,
                                            tadversary=tadversary,
                                            tgaug=tgaug,
                                            teps=0,
                                        )

                                        pt = pt.split("/")
                                        pt = pt[0 : len(pt) - 2]
                                        stdpt = ""
                                        for p in pt:
                                            stdpt += p + "/"
                                        stdpt += "res.csv"
                                        stddf = pd.read_csv(stdpt)
                                        mitdfs = []
                                        for lambda_index in lambda_indeces:
                                            pt = resacqobj.buildPath(
                                                basePath=basePath,
                                                standard=0,
                                                alg="DL",
                                                lamp=lamp,
                                                culture=culture,
                                                percent=percent,
                                                augment=augment,
                                                adversary=adv,
                                                lambda_index=lambda_index,
                                                taugment=taugment,
                                                tadversary=tadversary,
                                                tgaug=tgaug,
                                                teps=0,
                                            )
                                            pt = pt.split("/")
                                            pt = pt[: len(pt) - 2]
                                            mitpt = ""
                                            for p in pt:
                                                mitpt += p + "/"
                                            pt = mitpt + "res.csv"
                                            df = pd.read_csv(pt)
                                            mitdfs.append(df)

                                        title, svpath = gen_title_plot(
                                            lamp,
                                            culture,
                                            percent,
                                            augment,
                                            adv,
                                            taugment,
                                            tadversary,
                                            tgaug,
                                            0,
                                        )
                                        plotandsave(
                                            stddf=stddf,
                                            mitdfs=mitdfs,
                                            plot=plot,
                                            title=title,
                                            save=True,
                                            path=svpath,
                                        )

                                if not taugment and tadversary:
                                    for teps in test_eps:
                                        pt = resacqobj.buildPath(
                                            basePath=basePath,
                                            standard=1,
                                            alg="DL",
                                            lamp=lamp,
                                            culture=culture,
                                            percent=percent,
                                            augment=augment,
                                            adversary=adv,
                                            lambda_index=0,
                                            taugment=taugment,
                                            tadversary=tadversary,
                                            tgaug=0,
                                            teps=teps,
                                        )
                                        pt = pt.split("/")
                                        pt = pt[0 : len(pt) - 2]
                                        stdpt = ""
                                        for p in pt:
                                            stdpt += p + "/"
                                        stdpt += "res.csv"
                                        stddf = pd.read_csv(stdpt)
                                        mitdfs = []
                                        for lambda_index in lambda_indeces:
                                            pt = resacqobj.buildPath(
                                                basePath=basePath,
                                                standard=0,
                                                alg="DL",
                                                lamp=lamp,
                                                culture=culture,
                                                percent=percent,
                                                augment=augment,
                                                adversary=adv,
                                                lambda_index=lambda_index,
                                                taugment=taugment,
                                                tadversary=tadversary,
                                                tgaug=0,
                                                teps=teps,
                                            )
                                            pt = pt.split("/")
                                            pt = pt[: len(pt) - 2]
                                            mitpt = ""
                                            for p in pt:
                                                mitpt += p + "/"
                                            pt = mitpt + "res.csv"
                                            df = pd.read_csv(pt)
                                            mitdfs.append(df)

                                        title, svpath = gen_title_plot(
                                            lamp,
                                            culture,
                                            percent,
                                            augment,
                                            adv,
                                            taugment,
                                            tadversary,
                                            0,
                                            teps,
                                        )
                                        plotandsave(
                                            stddf=stddf,
                                            mitdfs=mitdfs,
                                            plot=plot,
                                            title=title,
                                            save=True,
                                            path=svpath,
                                        )

                                if (not taugment) and (not tadversary):
                                    pt = resacqobj.buildPath(
                                        basePath=basePath,
                                        standard=1,
                                        alg="DL",
                                        lamp=lamp,
                                        culture=culture,
                                        percent=percent,
                                        augment=augment,
                                        adversary=adv,
                                        lambda_index=0,
                                        taugment=taugment,
                                        tadversary=tadversary,
                                        tgaug=0,
                                        teps=0,
                                    )
                                    pt = pt.split("/")
                                    pt = pt[0 : len(pt) - 2]
                                    stdpt = ""
                                    for p in pt:
                                        stdpt += p + "/"
                                    stdpt += "res.csv"
                                    stddf = pd.read_csv(stdpt)
                                    mitdfs = []
                                    for lambda_index in lambda_indeces:
                                        pt = resacqobj.buildPath(
                                            basePath=basePath,
                                            standard=0,
                                            alg="DL",
                                            lamp=lamp,
                                            culture=culture,
                                            percent=percent,
                                            augment=augment,
                                            adversary=adv,
                                            lambda_index=lambda_index,
                                            taugment=taugment,
                                            tadversary=tadversary,
                                            tgaug=0,
                                            teps=0,
                                        )
                                        pt = pt.split("/")
                                        pt = pt[: len(pt) - 2]
                                        mitpt = ""
                                        for p in pt:
                                            mitpt += p + "/"
                                        pt = mitpt + "res.csv"
                                        df = pd.read_csv(pt)
                                        mitdfs.append(df)
                                    title, svpath = gen_title_plot(
                                        lamp,
                                        culture,
                                        percent,
                                        augment,
                                        adv,
                                        taugment,
                                        tadversary,
                                        0,
                                        0,
                                    )
                                    plotandsave(
                                        stddf=stddf,
                                        mitdfs=mitdfs,
                                        plot=plot,
                                        title=title,
                                        save=True,
                                        path=svpath,
                                    )


def get_best_df_gamma(mitdfs, tau=0.1):
    errs = []
    cics = []
    n = int(len(mitdfs) * tau)
    for i, mitdf in enumerate(mitdfs):
        err = perstr2float(mitdf["ERR"][0])
        errs.append([err, i])
        cic = perstr2float(mitdf["CIC"][0])
        cics.append(cic)

    errs = np.asarray(errs, dtype=object)
    errs = np.sort(errs, axis=0)
    errs = errs[:n]

    # tempcics = cics[:,1]

    tempcics = [cics[i] for i in errs[:, 1]]

    cicstar = min(tempcics)
    idx = cics.index(cicstar)

    return mitdfs[idx]


def get_test_name(taug, tadv, gaug=0, geps=0):
    if taug:
        if tadv:
            name = f"AUG={gaug}, ADV={geps}"
        else:
            name = f"AUG={gaug}"
    else:
        if tadv:
            name = f"ADV={geps}"
        else:
            name = f"NOAUG"

    return name


class Res2TabClass:

    def convert2list(self, test_name, std, df):
        res = [
            test_name,
            std,
            perstr2float(df["ERR^CULTURE 0"][0]),
            perstr2float(df["ERR^CULTURE 0 std"][0]),
            perstr2float(df["ERR^CULTURE 1"][0]),
            perstr2float(df["ERR^CULTURE 1 std"][0]),
            perstr2float(df["ERR^CULTURE 2"][0]),
            perstr2float(df["ERR^CULTURE 2 std"][0]),
            perstr2float(df["ERR"][0]),
            perstr2float(df["ERR std"][0]),
            perstr2float(df["CIC"][0]),
            perstr2float(df["CIC std"][0]),
        ]
        return res

    def get_path(self, base, lamp, culture, percent, aug, adv):
        pt = base
        if lamp:
            if culture == 0:
                pt += "LC/"
            if culture == 1:
                pt += "LF/"
            if culture == 2:
                pt += 'LT/'
        else:
            if culture == 0:
                pt += "CI/"
            if culture == 1:
                pt += "CJ/"
            if culture == 2:
                pt += 'CS/'

        pt += str(percent) + '/'

        if aug:
            if adv:
                pt+= 'TOTAUG/'
            else:
                pt += 'AUG/'
        else:
            if adv:
                pt += 'ADV/'
            else:
                pt += 'NOAUG/'

        return pt

    def conversion(self):
        def get_name_column(lamp, culture, percent, aug, adv):
            ## Returns the name of tab and the columns
            if lamp:
                columns = {
                    "TestSet": [],
                    "Mitigation": [],
                    "ERR^LC": [],
                    "ERR^LC std": [],
                    "ERR^LF": [],
                    "ERR^LF std": [],
                    "ERR^LT": [],
                    "ERR^LT std": [],
                    "ERR": [],
                    "ERR std": [],
                    "CIC": [],
                    "CIC std": [],
                }
                if culture == 0:
                    name = "LC"
                if culture == 1:
                    name = "LF"
                if culture == 2:
                    name = "LT"
            else:
                columns = {
                    "TestSet": [],
                    "Mitigation": [],
                    "ERR^CI": [],
                    "ERR^CI std": [],
                    "ERR^CJ": [],
                    "ERR^CJ std": [],
                    "ERR^CS": [],
                    "ERR^CS std": [],
                    "ERR": [],
                    "ERR std": [],
                    "CIC": [],
                    "CIC std": [],
                }
                if culture == 0:
                    name = "CI"
                if culture == 1:
                    name = "CJ"
                if culture == 2:
                    name = "CS"

            name += f" with p_u={percent},"
            if aug:
                if adv:
                    name += f"TOTAUG in Train"
                else:
                    name += f"AUG in Train"

            else:
                if adv:
                    name += f"ADV in Train"
                else:
                    name += f"NOAUG in Train"

            return name, columns

        # Structure of Tab is:
        # Per each DS
        # Per each Culture
        # Per each Percentage
        # Per each Data Augmentation in Training
        #   TestSet Mitigation  ERR^CULTURE 0 ERR^CULTURE 0 std,ERR^CULTURE 1,ERR^CULTURE 1 std,ERR^CULTURE 2,ERR^CULTURE 2 std,ERR,ERR std,CIC,CIC std
        #   NOAUG   STD         value   value   value   value   value ..
        #   NOAUG   MIT         value   value   value   value   value ...
        #   AUG=..
        # ...
        # Using TAU=0.1, we have to select the right LAMBDA

        basePath = "../Results/"

        resacqobj = ResAcquisitionClass()

        alg = "DL"
        lamps = [0, 1]
        cultures = [0, 1, 2]
        percents = [0.05, 0.1]
        augments = [0, 1]
        adversary = [0, 1]

        taugments = [0, 1]
        tadversaries = [0, 1]
        test_g_augs = [0.01, 0.05, 0.1]
        test_eps = [0.0005, 0.001, 0.005]

        lambda_indeces = range(0, 13)
        t_cults = [0, 1, 2]
        for lamp in lamps:
            for culture in cultures:
                for percent in percents:
                    for adv in adversary:
                        for aug in augments:
                            # Here we have to divide the DataFrames
                            name, columns = get_name_column(
                                lamp, culture, percent, aug, adv
                            )

                            df = pd.DataFrame(columns=columns)
                            for taug in taugments:
                                for tadv in tadversaries:
                                    if taug and tadv:
                                        for tgaug in test_g_augs:
                                            for teps in test_eps:
                                                pt = resacqobj.buildPath(
                                                    basePath=basePath,
                                                    standard=1,
                                                    alg="DL",
                                                    lamp=lamp,
                                                    culture=culture,
                                                    percent=percent,
                                                    augment=aug,
                                                    adversary=adv,
                                                    lambda_index=0,
                                                    taugment=taug,
                                                    tadversary=tadv,
                                                    tgaug=tgaug,
                                                    teps=teps,
                                                )
                                                pt = pt.split("/")
                                                pt = pt[0 : len(pt) - 2]
                                                stdpt = ""
                                                for p in pt:
                                                    stdpt += p + "/"
                                                stdpt += "res.csv"
                                                stddf = pd.read_csv(stdpt)
                                                mitdfs = []
                                                for lambda_index in lambda_indeces:
                                                    pt = resacqobj.buildPath(
                                                        basePath=basePath,
                                                        standard=0,
                                                        alg="DL",
                                                        lamp=lamp,
                                                        culture=culture,
                                                        percent=percent,
                                                        augment=aug,
                                                        adversary=adv,
                                                        lambda_index=lambda_index,
                                                        taugment=taug,
                                                        tadversary=tadv,
                                                        tgaug=tgaug,
                                                        teps=teps,
                                                    )
                                                    pt = pt.split("/")
                                                    pt = pt[: len(pt) - 2]
                                                    mitpt = ""
                                                    for p in pt:
                                                        mitpt += p + "/"
                                                    pt = mitpt + "res.csv"
                                                    tempdf = pd.read_csv(pt)
                                                    mitdfs.append(tempdf)

                                                mitdf = get_best_df_gamma(mitdfs)
                                                test_name = get_test_name(
                                                    taug, tadv, tgaug, teps
                                                )
                                                df.loc[len(df)] = self.convert2list(
                                                    test_name, "STD", stddf
                                                )
                                                df.loc[len(df)] = self.convert2list(
                                                    test_name, "MIT", mitdf
                                                )

                                    if taug and not tadv:
                                        for tgaug in test_g_augs:
                                            pt = resacqobj.buildPath(
                                                basePath=basePath,
                                                standard=1,
                                                alg="DL",
                                                lamp=lamp,
                                                culture=culture,
                                                percent=percent,
                                                augment=aug,
                                                adversary=adv,
                                                lambda_index=0,
                                                taugment=taug,
                                                tadversary=tadv,
                                                tgaug=tgaug,
                                                teps=0,
                                            )

                                            pt = pt.split("/")
                                            pt = pt[0 : len(pt) - 2]
                                            stdpt = ""
                                            for p in pt:
                                                stdpt += p + "/"
                                            stdpt += "res.csv"
                                            stddf = pd.read_csv(stdpt)
                                            mitdfs = []
                                            for lambda_index in lambda_indeces:
                                                pt = resacqobj.buildPath(
                                                    basePath=basePath,
                                                    standard=0,
                                                    alg="DL",
                                                    lamp=lamp,
                                                    culture=culture,
                                                    percent=percent,
                                                    augment=aug,
                                                    adversary=adv,
                                                    lambda_index=lambda_index,
                                                    taugment=taug,
                                                    tadversary=tadv,
                                                    tgaug=tgaug,
                                                    teps=0,
                                                )
                                                pt = pt.split("/")
                                                pt = pt[: len(pt) - 2]
                                                mitpt = ""
                                                for p in pt:
                                                    mitpt += p + "/"
                                                pt = mitpt + "res.csv"
                                                tempdf = pd.read_csv(pt)
                                                mitdfs.append(tempdf)

                                            mitdf = get_best_df_gamma(mitdfs)
                                            test_name = get_test_name(
                                                taug, tadv, tgaug, 0
                                            )
                                            df.loc[len(df)] = self.convert2list(
                                                test_name, "STD", stddf
                                            )
                                            df.loc[len(df)] = self.convert2list(
                                                test_name, "MIT", mitdf
                                            )

                                    if not taug and tadv:
                                        for teps in test_eps:
                                            pt = resacqobj.buildPath(
                                                basePath=basePath,
                                                standard=1,
                                                alg="DL",
                                                lamp=lamp,
                                                culture=culture,
                                                percent=percent,
                                                augment=aug,
                                                adversary=adv,
                                                lambda_index=0,
                                                taugment=taug,
                                                tadversary=tadv,
                                                tgaug=0,
                                                teps=teps,
                                            )
                                            pt = pt.split("/")
                                            pt = pt[0 : len(pt) - 2]
                                            stdpt = ""
                                            for p in pt:
                                                stdpt += p + "/"
                                            stdpt += "res.csv"
                                            stddf = pd.read_csv(stdpt)
                                            mitdfs = []
                                            for lambda_index in lambda_indeces:
                                                pt = resacqobj.buildPath(
                                                    basePath=basePath,
                                                    standard=0,
                                                    alg="DL",
                                                    lamp=lamp,
                                                    culture=culture,
                                                    percent=percent,
                                                    augment=aug,
                                                    adversary=adv,
                                                    lambda_index=lambda_index,
                                                    taugment=taug,
                                                    tadversary=tadv,
                                                    tgaug=0,
                                                    teps=teps,
                                                )
                                                pt = pt.split("/")
                                                pt = pt[: len(pt) - 2]
                                                mitpt = ""
                                                for p in pt:
                                                    mitpt += p + "/"
                                                pt = mitpt + "res.csv"
                                                tempdf = pd.read_csv(pt)
                                                mitdfs.append(tempdf)

                                            mitdf = get_best_df_gamma(mitdfs)
                                            test_name = get_test_name(
                                                taug, tadv, 0, teps
                                            )
                                            df.loc[len(df)] = self.convert2list(
                                                test_name, "STD", stddf
                                            )
                                            df.loc[len(df)] = self.convert2list(
                                                test_name, "MIT", mitdf
                                            )

                                    if (not taug) and (not tadv):
                                        pt = resacqobj.buildPath(
                                            basePath=basePath,
                                            standard=1,
                                            alg="DL",
                                            lamp=lamp,
                                            culture=culture,
                                            percent=percent,
                                            augment=aug,
                                            adversary=adv,
                                            lambda_index=0,
                                            taugment=taug,
                                            tadversary=tadv,
                                            tgaug=0,
                                            teps=0,
                                        )
                                        pt = pt.split("/")
                                        pt = pt[0 : len(pt) - 2]
                                        stdpt = ""
                                        for p in pt:
                                            stdpt += p + "/"
                                        stdpt += "res.csv"
                                        stddf = pd.read_csv(stdpt)
                                        mitdfs = []
                                        for lambda_index in lambda_indeces:
                                            pt = resacqobj.buildPath(
                                                basePath=basePath,
                                                standard=0,
                                                alg="DL",
                                                lamp=lamp,
                                                culture=culture,
                                                percent=percent,
                                                augment=aug,
                                                adversary=adv,
                                                lambda_index=lambda_index,
                                                taugment=taug,
                                                tadversary=tadv,
                                                tgaug=0,
                                                teps=0,
                                            )
                                            pt = pt.split("/")
                                            pt = pt[: len(pt) - 2]
                                            mitpt = ""
                                            for p in pt:
                                                mitpt += p + "/"
                                            pt = mitpt + "res.csv"
                                            tempdf = pd.read_csv(pt)
                                            mitdfs.append(tempdf)

                                        mitdf = get_best_df_gamma(mitdfs)
                                        # print(f"df is {df}")
                                        # print(f"mit df is {mitdf}")
                                        test_name = get_test_name(taug, tadv, 0, 0)
                                        # print(f"test name is {test_name}")
                                        # print(f"len of df is {len(df)}")
                                        # print(f"columns of df is {df.columns}")
                                        # print(f"list of values is {self.convert2list(test_name, 'STD',  stddf)}")
                                        # print(f"len of df columns is {len(df.columns)}")
                                        # print(f"len of list is {len(self.convert2list(test_name, 'STD',  stddf))}")
                                        df.loc[len(df)] = self.convert2list(
                                            test_name, "STD", stddf
                                        )

                                        df.loc[len(df)] = self.convert2list(
                                            test_name, "MIT", mitdf
                                        )
                            print(name)
                            print(df)
                            pt = self.get_path('../Results/TABRES/', lamp, culture, percent, aug, adv)
                            fileObj =  FileManagerClass(pt)
                            df.to_csv(pt + 'df')

                            


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
    lambda_indeces = range(0, 13)
    taugments = [0, 1]
    tadversaries = [0, 1]
    test_g_augs = [0.01, 0.05, 0.1]
    test_eps = [0.0005, 0.001, 0.005]
    t_cults = [0, 1, 2]

    # print_tables(standards, lamps, cultures, percents, augments, adversary, lambda_indeces, taugments, tadversaries, test_g_augs, test_eps, paths, visobj)
    # graphic_lambdas_comparison(standards, lamps, cultures, percents, augments, adversary, lambda_indeces, taugments, tadversaries, test_g_augs, test_eps, resacqobj)

    res2tabObj = Res2TabClass()
    res2tabObj.conversion()


if __name__ == "__main__":
    main()
