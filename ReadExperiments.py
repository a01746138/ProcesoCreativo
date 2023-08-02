# Create three identified matrices of the experiments
# results performed in MATLAB
# ==========================================================

import numpy as np
import os
import pandas as pd


def join():
    folder = os.listdir('experiments\\')
    pop = []

    for file in folder:
        f = np.loadtxt('experiments\\' + file, delimiter=',')
        for ind in f:
            pop.append(ind)
    pop = np.array(pop)

    # print(f'({len(pop)},{len(pop[0])})')

    return pd.DataFrame(pop, columns=['Vbatt', 'Qbatt', 'Ndiff',
                                      'Rwheel', 'MaxPmot', 'Mass',
                                      'Hcons', 'Pmech'])


def separate():
    pop = join()
    data = pop[(pop['Hcons'] != 0) & (pop['Hcons'] != 1)]
    err = pop[(pop['Hcons'] == 0)]
    over = pop[(pop['Hcons'] == 1)]

    return data, err, over
