# Visualize the mean and std from the data of the cross validation step to
# select the surrogate model that better fits the MATLAB model
# ==========================================================


import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np


df = pd.DataFrame(-np.loadtxt('CrossValidation.txt'),
                  columns=['ann_Hcons', 'ann_Pmech',
                           'svr_Hcons', 'svr_Pmech',
                           'dtr_Hcons', 'dtr_Pmech',
                           'rfr_Hcons', 'rfr_Pmech'])


# sn.boxplot(data=pd.melt(df[['ann_Hcons', 'svr_Hcons', 'dtr_Hcons', 'rfr_Hcons']]),
#            x='variable', y='value')
# plt.ylabel('MSE')
# plt.xlabel('Hydrogen consumption surrogate models')
# plt.tight_layout()
# plt.subplots_adjust(top=0.88)
# plt.show()

sn.boxplot(data=pd.melt(df[['ann_Pmech', 'svr_Pmech', 'dtr_Pmech', 'rfr_Pmech']]),
           x='variable', y='value')
plt.ylabel('MSE')
plt.xlabel('Total mechanical power of the motor surrogate models')
plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.show()

print(df.mean())
print(df.std())
