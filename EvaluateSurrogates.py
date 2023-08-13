import joblib
import pandas as pd
import numpy as np
from ProcessingData import normalization

data = pd.read_csv(filepath_or_buffer='Data.csv')
df = normalization(data).drop(['Hcons', 'Pmech'], axis=1)

print(normalization(data)['Hcons'].mean())
# print(data['Hcons'])

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

models = [ann_h, ann_mech,
          svr_h, svr_mech,
          dtr_h, dtr_mech,
          rfr_h, rfr_mech]
models_lbl = ['ann_h', 'ann_mech',
              'svr_h', 'svr_mech',
              'dtr_h', 'dtr_mech',
              'rfr_h', 'rfr_mech']

zlist = df.shape[0] * [0]

for lbl in models_lbl:
    data[lbl] = zlist

# for m in models:
#     plist = []
#     for index, row in df.iterrows():
#         xx = np.reshape([row['Vbatt'], row['Qbatt'], row['Ndiff'],
#                          row['Rwheel'], row['MaxPmot'], row['Mass']], (1, -1))
#         pred = m.predict(xx)
#         plist.append(pred)
#

