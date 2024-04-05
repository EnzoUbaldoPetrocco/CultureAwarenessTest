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


def main():
    visobj = VisualizerClass()
    resacqobj = ResAcquisitionClass()

    paths = resacqobj.get_paths(basePath="../Results/")

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

    for standard in standards:
        for lamp in lamps:
            for culture in cultures:
                for percent in range(len(percents)):
                    for augment in augments:
                        for adv in adversary:
                            if standard:
                                for taugment in taugments:
                                    for tadversary in tadversaries:
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
                            else:
                                for lambda_index in lambda_indeces:
                                    for taugment in taugments:
                                        for tadversary in tadversaries:
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
                                                    print(pt)
                                                    df = pd.read_csv(pt)
                                                    visobj.plot_table(
                                                        df[
                                                            df.columns[
                                                                1 : len(df.columns)
                                                            ]
                                                        ],
                                                        pt,
                                                    )


if __name__ == "__main__":
    main()
