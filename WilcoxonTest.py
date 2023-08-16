import pandas as pd
import numpy as np
import joblib
from scipy.stats import wilcoxon

data = pd.read_csv(filepath_or_buffer='Data02.csv')

h_models_lbl = ['Hcons', 'Hcons_ann', 'Hcons_svr', 'Hcons_dtr', 'Hcons_rfr']
mech_models_lbl = ['Pmech', 'Pmech_ann', 'Pmech_svr', 'Pmech_dtr', 'Pmech_rfr']


def wilcoxon_test(lbl):
    pmatrix = []
    for i in lbl:
        plist = []
        for j in lbl:
            if i != j:
                p = wilcoxon(data[i], data[j]).pvalue
            else:
                p = 1
            plist.append(str(int(p * 1e6) / 1e6))
        pmatrix.append(plist)

    return pd.DataFrame(pmatrix, columns=lbl, index=lbl)


def mse(lbl):
    m = []
    for i in lbl[1:]:
        if i in h_models_lbl:
            error = np.sum((data['Hcons'] - data[i]) ** 2)/data.shape[0]
        else:
            error = np.sum((data['Pmech'] - data[i]) ** 2)/data.shape[0]
        m.append(error)

    return pd.DataFrame(m, index=lbl[1:], columns=[lbl[0]])


print(wilcoxon_test(h_models_lbl))
print('--------------------')
print(wilcoxon_test(mech_models_lbl))
print('--------------------')
print(data[['Pmech', 'Pmech_dtr', 'Pmech_svr']])
print('--------------------')
print(data[['Hcons', 'Hcons_dtr', 'Hcons_ann']])
print('--------------------')

print(mse(h_models_lbl))
print('--------------------')
print(mse(mech_models_lbl))

