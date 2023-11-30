# from SMSEMOA import SMSEMOA
# from PMOP import MyPMOP
# from VisualizeFront import plot_front
# import time
#
# start_time = time.time()

# n_gen = 40000
# algorithm = SMSEMOA(pop_size=100, n_gen=n_gen,
#                     problem=MyPMOP(lambda_mass=1.0), verbose=True)
#
# pop, nds = algorithm()
#
# print(f'Time: {time.time() - start_time}')
#
# plot_front(nds['F'])

# =============================================================
# =============================================================

# from MOEAD import MOEAD
# from PMOP import MyPMOP
# from VisualizeFront import plot_front
# import time
#
# start_time = time.time()
#
# n_gen = 400
# algorithm = MOEAD(n_gen=n_gen, problem=MyPMOP(lambda_mass=0.0),
#                   pop_size=100, verbose=True)
#
# pop, nds = algorithm()
#
# print(f'Time to run the algorithm for {n_gen}: {time.time() - start_time}')
#
# plot_front(nds['F'])

# =============================================================
# =============================================================

from NSGA3 import NSGA3
from PMOP import MyPMOP
from VisualizeFront import plot_front
import time

start_time = time.time()

n_gen = 10
algorithm = NSGA3(n_gen=n_gen, problem=MyPMOP(lambda_mass=0.0),
                  pop_size=100, verbose=True)

pop, nds = algorithm()

print(f'Time to run the algorithm for {n_gen} generations: {time.time() - start_time}')

plot_front(nds['F'])

# =============================================================
# =============================================================

# from PMOEA import PMOEA
# from PMOP import MyPMOP
#
# algorithm = PMOEA(n_gen=10, pop_size=100, lambda_partitions=5,
#                   problem=MyPMOP, algorithm='nsga3', verbose=True)
#
# algorithm()

# =============================================================
# =============================================================

# from MOEAD import MOEAD
# from PMOP import MyPMOP
#
# n_gen = 10
# algorithm = MOEAD(pop_size=100, n_gen=n_gen,
#                   problem=MyPMOP(lambda_mass=0.0), verbose=True)
#
# pop, nds, hv = algorithm()
#
# print(nds)
#
# hv_df = pd.DataFrame(hv, columns=['n_gen', 'hv_value']).to_csv(path_or_buf='HVanalysis.csv')

# =============================================================
# =============================================================

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sb
#
#
# df = pd.read_csv(filepath_or_buffer='HVanalysis.csv')
#
# sb.lineplot(x=np.array(df['n_eval'])[3:], y=np.array(df['hv_value'])[3:])
# plt.xlabel('Number of function evaluations')
# plt.ylabel('Hypervolume value')
# plt.show()

# =============================================================
# =============================================================

# from IMIA import IMIA
# from PMOP import MyPMOP
# from VisualizeFront import plot_front
# import time
#
# start_time = time.time()
#
# n_gen = 60
# algorithm = IMIA(pop_size=100, n_gen=n_gen,
#                  problem=MyPMOP(lambda_mass=0.0), verbose=True,
#                  indicators=['HV', 'R2', 'EpsPlus', 'DeltaP', 'IGDPlus'])
#
# pop, nds, hv = algorithm()
#
# print(f'Time to run the algorithm for {n_gen} generations: {time.time() - start_time}')
#
# plot_front(nds['F'])

# =============================================================
# =============================================================

# from PIMIA import IMIA
# from PMOP_PIMIA import MyPMOP
# import time
#
# start_time = time.time()
#
# n_gen = 20
# algorithm = IMIA(pop_size=100, n_gen=n_gen,
#                  problem=MyPMOP(), verbose=True,
#                  indicators=['HV', 'R2', 'EpsPlus', 'DeltaP', 'IGDPlus'])
#
# pop, nds, hv = algorithm()
#
# print(f'Time to run the algorithm for {n_gen} generations: {time.time() - start_time}')
# print('================')
# print(hv)

# =============================================================
# =============================================================

# from PMOP import MyPMOP
# import time
#
# p = MyPMOP()
#
# start_time = time.time()
# print(p.evaluate([0, 0, 0, 0, 0, 0]))
# print(f'Time: {time.time() - start_time}')
#
# start_time = time.time()
# p.evaluate([0, 0, 0, 0, 0, 1])
# print(f'Time: {time.time() - start_time}')
#
# start_time = time.time()
# p.evaluate([0, 0, 0, 1, 0, 0])
# p.evaluate([1, 0, 0, 1, 0, 0])
# p.evaluate([0, 0.5, 0, 1, 0, 0])
# p.evaluate([0, 0, 0.5, 1, 0, 0])
# p.evaluate([0, 0, 0, 1, 0, 0])
# p.evaluate([0, 0, 0, 1, 0.5, 0])
# p.evaluate([0, 0.1, 0, 1, 0, 0])
# p.evaluate([0, 0, 0, 1, 0, 0])
# p.evaluate([0, 1, 0, 1, 0, 0])
# p.evaluate([0, 0, 0, 1, 0.6, 0])
# print(f'Time: {time.time() - start_time}')
