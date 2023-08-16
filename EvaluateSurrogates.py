import joblib
import pandas as pd
import numpy as np
from ProcessingData import normalization, decode

data = pd.read_csv(filepath_or_buffer='Data.csv')
df = normalization(data)  # .drop(['Hcons', 'Pmech'], axis=1)

data02 = pd.read_csv(filepath_or_buffer='Data02.csv')

ann_h = joblib.load(filename='surrogate\\ann_h_model.joblib',
                    mmap_mode='r')
ann_mech = joblib.load(filename='surrogate\\ann_mech_model.joblib',
                       mmap_mode='r')
svr_h = joblib.load(filename='surrogate\\svr_h_model.joblib',
                    mmap_mode='r')
svr_mech = joblib.load(filename='surrogate\\svr_mech_model.joblib',
                       mmap_mode='r')
dtr_h = joblib.load(filename='surrogate\\dtr_h_model.joblib',
                    mmap_mode='r')
dtr_mech = joblib.load(filename='surrogate\\dtr_mech_model.joblib',
                       mmap_mode='r')
rfr_h = joblib.load(filename='surrogate\\rfr_h_model.joblib',
                    mmap_mode='r')
rfr_mech = joblib.load(filename='surrogate\\rfr_mech_model.joblib',
                       mmap_mode='r')

# models = [ann_h, svr_h, dtr_h, rfr_h,
#           ann_mech, svr_mech, dtr_mech, rfr_mech]
# models_lbl = ['Hcons_ann', 'Hcons_svr', 'Hcons_dtr', 'Hcons_rfr',
#               'Pmech_ann', 'Pmech_svr', 'Pmech_dtr', 'Pmech_rfr']

models = [ann_h]
models_lbl = ['Hcons_ann']


c = 0
for m in models:
    plist = []
    for index, row in df.iterrows():
        xx = np.reshape([row['Vbatt'], row['Qbatt'], row['Ndiff'],
                         row['Rwheel'], row['MaxPmot'], row['Mass']], (1, -1))
        pred = m.predict(xx)
        plist.append(pred)
    data02[models_lbl[c]] = decode(pd.DataFrame(plist, columns=[models_lbl[c]]))
    c += 1

# data02.to_csv(path_or_buf='Data02.csv', index=False)

