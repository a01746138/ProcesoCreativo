# Visualize the data points of the ExperimentsMATLAB
# ==========================================================

from ReadExperiments import separate
from ProcessingData import normalization
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

# [Vbatt, Qbatt, Ndiff, Rwheel, MaxPmot, Mass, Hconsumed, Pmech]
data, err, over = separate()

# Plot the objective space of the MATLAB's data
# -----------------------------------------
sn.scatterplot(x=data['Hcons'][data['Mass'] == 1600.0],
               y=-data['Pmech'][data['Mass'] == 1600.0])
plt.xlabel('Hydrogen consumption [kg]')
plt.ylabel('Total mechanical power of the motor [kW]')
plt.xlim(0.3, 1.5)
plt.ylim(-100, -10)
plt.show()


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
# sn.scatterplot(x=data['Hcons'][data['Mass'] < 1650],
#            y=-data['Pmech'][data['Mass'] < 1650])
# sn.lineplot(x=f['Hcons'], y=-f['Pmech'], color='r')
# plt.xlabel('Hydrogen consumption [kg]')
# plt.ylabel('Total mechanical power of the motor [kW]')
# plt.title(r'Range of vehicle mass $<1650$ [kg]')
# plt.show()


# Distribution plots regarding each objective
# -----------------------------------------
# sn.displot(x=data['Hcons'])
# plt.xlabel('Hydrogen consumption [kg]')
# plt.ylabel('Count')
# plt.show()

# sn.displot(x=data['Pmech'])
# plt.xlabel('Total mechanical power of the motor [kW]')
# plt.ylabel('Count')
# plt.show()
