import numpy as np
import matplotlib.pyplot as plt


v = np.loadtxt(fname='Velocity.txt', delimiter=',')

data = []
for i in range(len(v)):
    data.append([i, v[i]])

data = np.array(data)

# plt.figure(dpi=300)
# plt.plot(data[:, 0], data[:, 1], linewidth=1)
# plt.xlabel('Time [s]')
# plt.ylabel('Velocity [m/s]')
# plt.title('Longitudinal drive cycle')

# plt.savefig(fname='DriveCycle.png')

# print(f'Velocity = {np.sum(v)}')