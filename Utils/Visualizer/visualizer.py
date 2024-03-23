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
        fig, ax = plt.subplots()
        # hide axes
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')
        tab = ax.table(cellText=df.values, colLabels=df.columns, loc='center')
        tab.auto_set_font_size(False)
        tab.set_fontsize(9.5)
        fig.tight_layout()
        ax.set_title(title)
        plt.show()


def main():
    visobj = VisualizerClass()
    resacqobj = ResAcquisitionClass()
    
    paths = resacqobj.get_paths(basePath="../Results/")
    for standard in range(len(paths)):
        for lamp in range(len(paths[standard])):
            for culture in range(len(paths[standard][lamp])):
                for percent in range(len(paths[standard][lamp][culture])):
                    for augment in range(
                        len(paths[standard][lamp][culture][percent])
                    ):
                        for adv in range(
                            len(paths[standard][lamp][culture][percent][augment])
                        ):
                            if standard:
                                for taugment in range(
                                    len(
                                        paths[standard][lamp][culture][percent][
                                            augment
                                        ][adv]
                                    )
                                ):
                                    for tadversary in range(
                                        len(
                                            paths[standard][lamp][culture][
                                                percent
                                            ][augment][adv][taugment]
                                        )
                                    ):
                                        for tgaug in range(
                                            len(
                                                paths[standard][lamp][culture][
                                                    percent
                                                ][augment][adv][taugment][
                                                    tadversary
                                                ]
                                            )
                                        ):
                                            for teps in range(
                                                len(
                                                    paths[standard][lamp][
                                                        culture
                                                    ][percent][augment][adv][
                                                        taugment
                                                    ][
                                                        tadversary
                                                    ][
                                                        tgaug
                                                    ]
                                                )
                                            ):

                                                pt = paths[standard][lamp][
                                                            culture
                                                        ][percent][augment][adv][
                                                            taugment
                                                        ][
                                                            tadversary
                                                        ][
                                                            tgaug
                                                        ][
                                                            teps
                                                        ][0]
                                                
                                                tempst = pt.split("/")
                                                tempst2 = ""
                                                for i in range(
                                                    len(tempst) - 2
                                                ):
                                                    tempst2 += (
                                                        tempst[i] + "/"
                                                    )
                                                pt = tempst2
                                                df = pd.read_csv(pt)
                                                visobj.plot_table(df[df.columns[1:9]], pt)
                                                
                            else:
                                for lambda_index in range(
                                    len(
                                        paths[standard][lamp][culture][percent][
                                            augment
                                        ][adv]
                                    )
                                ):
                                    for taugment in range(
                                        len(
                                            paths[standard][lamp][culture][
                                                percent
                                            ][augment][adv][lambda_index]
                                        )
                                    ):
                                        for tadversary in range(
                                            len(
                                                paths[standard][lamp][culture][
                                                    percent
                                                ][augment][adv][lambda_index][
                                                    taugment
                                                ]
                                            )
                                        ):
                                            for tgaug in range(
                                                len(
                                                    paths[standard][lamp][
                                                        culture
                                                    ][percent][augment][adv][
                                                        lambda_index
                                                    ][
                                                        taugment
                                                    ][
                                                        tadversary
                                                    ]
                                                )
                                            ):
                                                for teps in range(
                                                    len(
                                                        paths[standard][lamp][
                                                            culture
                                                        ][percent][augment][adv][
                                                            lambda_index
                                                        ][
                                                            taugment
                                                        ][
                                                            tadversary
                                                        ][
                                                            tgaug
                                                        ]
                                                    )
                                                ):
                                                    
                                                    pt = paths[standard][
                                                                lamp
                                                            ][culture][percent][
                                                                augment
                                                            ][
                                                                adv
                                                            ][
                                                                lambda_index
                                                            ][
                                                                taugment
                                                            ][
                                                                tadversary
                                                            ][
                                                                tgaug
                                                            ][
                                                                teps
                                                            ][0]
                                                    
                                                    tempst = pt.split("/")
                                                    tempst2 = ""
                                                    for i in range(
                                                        len(tempst) - 2
                                                    ):
                                                        tempst2 += (
                                                            tempst[i] + "/"
                                                        )
                                                    pt = tempst2 + "res.csv"
                                                    df = pd.read_csv(pt)
                                                    visobj.plot_table(df[df.columns[1:9]], pt)

    
        
    



if __name__ == "__main__":
    main()




