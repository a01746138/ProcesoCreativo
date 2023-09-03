import joblib
import pandas as pd
import numpy as np
from ExperimentsMATLAB.ProcessingData import normalization, decode

data = pd.read_csv(filepath_or_buffer='../ExperimentsMATLAB/Data.csv')
df = normalization(data)  # .drop(['Hcons', 'Pmech'], axis=1)

# data02 = data
data02 = pd.read_csv(filepath_or_buffer='Data02.csv')

model_h_ann = joblib.load(filename='model_h_ann.joblib',
                          mmap_mode='r')
model_mech_ann = joblib.load(filename='model_mech_ann.joblib',
                             mmap_mode='r')
model_h_svr = joblib.load(filename='model_h_svr.joblib',
                          mmap_mode='r')
model_mech_svr = joblib.load(filename='model_mech_svr.joblib',
                             mmap_mode='r')
model_h_dtr = joblib.load(filename='model_h_dtr.joblib',
                          mmap_mode='r')
model_mech_dtr = joblib.load(filename='model_mech_dtr.joblib',
                             mmap_mode='r')
model_h_rfr = joblib.load(filename='model_h_rfr.joblib',
                          mmap_mode='r')
model_mech_rfr = joblib.load(filename='model_mech_rfr.joblib',
                             mmap_mode='r')

models = [model_h_ann, model_h_svr, model_h_dtr, model_h_rfr,
          model_mech_ann, model_mech_svr, model_mech_dtr, model_mech_rfr]
models_lbl = ['Hcons_ann', 'Hcons_svr', 'Hcons_dtr', 'Hcons_rfr',
              'Pmech_ann', 'Pmech_svr', 'Pmech_dtr', 'Pmech_rfr']

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

data02.to_csv(path_or_buf='Data02.csv', index=False)

