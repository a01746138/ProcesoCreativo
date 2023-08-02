# Visualize the data points of the experiments
# ==========================================================

from ReadExperiments import separate
from ProcessingData import normalization
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

# [Vbatt, Qbatt, Ndiff, Rwheel, MaxPmot, Mass, Hconsumed, Pmech]
data, err, over = separate()

f = pd.DataFrame(np.loadtxt('runs\\z_sms_lam0.0_exp0.txt', delimiter=','),
                 columns=['Vbatt', 'Qbatt', 'Ndiff',
                          'Rwheel', 'MaxPmot', 'Mass',
                          'Hcons', 'Pmech'])

# Plot the objective space of the MATLAB's data
# -----------------------------------------
# sn.scatterplot(x=data['Hcons'][data['Hcons'] > 0.46],
#                y=data['Pmech'][data['Hcons'] > 0.46])
# plt.xlabel('Hydrogen consumption [kg]')
# plt.ylabel('Total mechanical power of the motor [kW]')
# plt.show()


# sn.displot(x=data['Hcons'][(data['Mass'] < 1800) & (data['Mass'] > 1700)],
#            y=-data['Pmech'][(data['Mass'] < 1800) & (data['Mass'] > 1700)])
# plt.xlabel('Hydrogen consumption [kg]')
# plt.ylabel('Total mechanical power of the motor [kW]')
# plt.title(r'Vehicle mass $<1700$ [kg]')
# plt.tight_layout()
# plt.subplots_adjust(top=0.88)
# plt.show()


# Plot the data distribution of errors to find patterns
# -----------------------------------------
# sn.pairplot(
#     data=err[err['Vbatt'] > 250],
#     x_vars=['Vbatt', 'Qbatt', 'Ndiff', 'Rwheel', 'MaxPmot'],
#     y_vars='Mass',
#     kind='hist')
# plt.show()


# -----------------------------------------
# sn.scatterplot(x=data['Hcons'],
#                y=-data['Pmech'])
sn.lineplot(x=f['Hcons'], y=f['Pmech'], color='r')
plt.xlabel('Hydrogen consumption [kg]')
plt.ylabel('Total mechanical power of the motor [kW]')
plt.show()
