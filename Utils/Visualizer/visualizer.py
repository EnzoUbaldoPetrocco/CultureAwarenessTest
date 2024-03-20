from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

class VisualizerClass:
    def plot_table(self, df):
        fig, ax = plt.subplots()
        # hide axes
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')
        ax.table(cellText=df.values, colLabels=df.columns, loc='center')
        fig.tight_layout()
        plt.show()


def main():
    df = pd.DataFrame(np.random.randn(10, 4), columns=list('ABCD'))
    visclas = VisualizerClass()
    visclas.plot_table(df)



if __name__ == "__main__":
    main()




