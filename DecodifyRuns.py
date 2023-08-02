import os
import numpy as np
import pandas as pd
from ReadExperiments import separate

folder = os.listdir('runs\\')

a, b, c = separate()

for file in folder:
    if file[0] != 'z':
        f = pd.DataFrame(np.loadtxt('runs\\' + file, delimiter=','),
                         columns=['Vbatt', 'Qbatt', 'Ndiff',
                                  'Rwheel', 'MaxPmot', 'Mass',
                                  'Hcons', 'Pmech'])
        for col in f.columns:
            f[col] = f[col] * (a[col].max() - a[col].min()) + a[col].min()

        np.savetxt(fname=f'runs\\z_{file}',
                   X=np.array(np.array(f)),
                   delimiter=',')
