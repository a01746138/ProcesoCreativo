import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import numpy as np


def plot_front(front):
    data = pd.read_csv('Data.csv')
    h_min = data['Hcons'].min()
    h_max = data['Hcons'].max()
    mech_min = data['Pmech'].min()
    mech_max = data['Pmech'].max()

    x = np.array(front)[:, 0] * (h_max - h_min) + h_min
    y = - (abs(np.array(front)[:, 1]) * (mech_max - mech_min) + mech_min)
    sb.scatterplot(x=x, y=y)
    plt.xlabel('Hydrogen consumption [kg]')
    plt.ylabel('Total mechanical power of the motor [kW]')
    plt.xlim(h_min, h_max)
    plt.ylim(-mech_max, -mech_min)
    plt.show()

