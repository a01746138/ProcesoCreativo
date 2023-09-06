import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import numpy as np


def plot_front(front, type='scr'):
    data = pd.read_csv('../ExperimentsMATLAB/Data.csv')
    h_min = data['Hcons'].min()
    h_max = data['Hcons'].max()
    mech_min = data['Pmech'].min()
    mech_max = data['Pmech'].max()

    x = np.array(front)[:, 0] * (h_max - h_min) + h_min
    y = - (abs(np.array(front)[:, 1]) * (mech_max - mech_min) + mech_min)

    if type == 'scr':
        sb.scatterplot(x=x, y=y)
    elif type == 'line':
        sb.lineplot(x=x, y=y)
    else:
        return 'Type of plot not defined.'

    plt.xlabel('Hydrogen consumption [kg]')
    plt.ylabel('Total mechanical power of the motor [kW]')
    plt.xlim(0.3, 1.5)
    plt.ylim(-100, -10)
    plt.show()


