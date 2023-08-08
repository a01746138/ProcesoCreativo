import numpy as np
import os
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from pymoo.indicators.hv import HV
from RunAlgorithm import RunAlgorithm


# gen = [10, 20, 30, 40, 50, 60, 70, 80, 90,
#        100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
#        1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
#
# alg = RunAlgorithm(lam=[0.0], n_gen=gen, rand=False)
# alg.sms_evaluate()


def calc_hv():
    path = 'C:\\Users\\luiz4\\PycharmProjects\\ProcesoCreativo\\'
    folder = os.listdir(path + 'hv_runs\\')
    p = []

    for file in folder:
        n = ''
        count = 14
        n_exp = ''
        while n != '_':
            n_exp += n
            count += 1
            n = file[count]

        f = np.loadtxt(path + 'hv_runs\\' + file, delimiter=',')
        df = pd.DataFrame(f, columns=['Vbatt', 'Qbatt', 'Ndiff',
                                      'Rwheel', 'MaxPmot', 'Mass',
                                      'Hcons', 'Pmech'])

        ref_point = np.array([df['Hcons'].max(), df['Pmech'].max()])
        front = np.array(df[['Hcons', 'Pmech']])

        ind = HV(ref_point=ref_point)

        p.append([int(n_exp), ind(front)])
        # print(f'{n_exp}: {ind(front)}')

    return pd.DataFrame(p, columns=['n_gen', 'hv'])


def plot_hv(values):
    sn.lineplot(x=values['n_gen'], y=values['hv'])
    # plt.xscale('log')
    plt.xlabel('Number of generations')
    plt.ylabel('Hypervolume value')
    plt.show()


plot_hv(calc_hv())
