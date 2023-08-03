import joblib
import numpy as np
import pandas as pd

from ReadExperiments import separate
from ProcessingData import normalization

a, b, c = separate()

f = normalization(a)

df = np.array(f.drop(['Hcons', 'Pmech'], axis=1))

h_model = joblib.load(filename='surrogate\\' + 'ann_h_model.joblib',
                      mmap_mode='r')
mech_model = joblib.load(filename='surrogate\\' + 'svr_mech_model.joblib',
                         mmap_mode='r')

m = []
for ind in df[0:10]:
    x = np.reshape([ind[0], ind[1], ind[2], ind[3], ind[4], ind[5]], (1, -1))
    f1 = h_model.predict(x)
    f2 = mech_model.predict(x)
    m.append([ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], f1[0], f2[0]])

print(f.head(10))
print(pd.DataFrame(m))
