# Create three identified matrices of the experiments
# results performed in MATLAB
# =========================================================

import numpy as np
import os
import pandas as pd


def join():
    path = 'C:\\Users\\luiz4\\PycharmProjects\\ProcesoCreativo\\'
    folder = os.listdir(path + 'experiments\\')
    pop = []

    for file in folder:
        if file[0:2] == 'id' or file[0:6] == 'lambda':
            f = np.loadtxt(path + 'experiments\\' + file, delimiter=',')
            for ind in f:
                pop.append(ind)
    pop = np.array(pop)

    # print(f'({len(pop)},{len(pop[0])})')

    return pd.DataFrame(pop, columns=['Vbatt', 'Qbatt', 'Ndiff',
                                      'Rwheel', 'MaxPmot', 'Mass',
                                      'Hcons', 'Pmech'])


def separate():
    pop = join()
    data = pop[(pop['Hcons'] != 0) & (pop['Hcons'] != 1)].reset_index(drop=True)
    err = pop[(pop['Hcons'] == 0)].reset_index(drop=True)
    over = pop[(pop['Hcons'] == 1)].reset_index(drop=True)

    return data, err, over


def save_csv():
    # Create the Data.csv file
    a, b, c = separate()
    a.to_csv(path_or_buf='Data.csv', index=False)
    print(a.shape)


save_csv()
