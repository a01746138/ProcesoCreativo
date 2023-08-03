import numpy as np
import pandas as pd
from ReadExperiments import separate
from ProcessingData import normalization


def extra_outliers():
    data, err, over = separate()

    f = normalization(data)
    a = []

    for i in data[data['Hcons'] < 0.46].index:
        a.append(np.array(f.iloc[i]))

    print(pd.DataFrame(a))

    exp = []
    for j in a:
        for _ in range(100):
            ind = []
            for k in j[0:6]:
                eps = np.random.uniform(-0.01, 0.01)
                value = k + eps
                ind.append(value)
            exp.append(ind)

    exp = pd.DataFrame(exp, columns=['Vbatt', 'Qbatt', 'Ndiff',
                                     'Rwheel', 'MaxPmot', 'Mass'])

    for col in exp.columns:
        exp[col] = exp[col] * (data[col].max() - data[col].min()) + data[col].min()

    return exp


# f = pd.DataFrame(np.loadtxt('runs\\' + 'z_sms_lam0.0_ngen30000_exp0.txt', delimiter=','),
#                  columns=['Vbatt', 'Qbatt', 'Ndiff',
#                           'Rwheel', 'MaxPmot', 'Mass',
#                           'Hcons', 'Pmech'])
#
# exp = f.drop(['Hcons', 'Pmech'], axis=1)

# np.savetxt('experiments\\extra02.txt', X=np.array(extra_outliers()), delimiter=',')
