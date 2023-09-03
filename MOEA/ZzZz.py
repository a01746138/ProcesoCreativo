from SMSEMOA import SMSEMOA
from PMOP import MyPMOP
from VisualizeFront import plot_front
import time

start_time = time.time()
algorithm = SMSEMOA(pop_size=100, n_gen=4000, problem=MyPMOP(lambda_mass=0.0), verbose=True)

pop, nds = algorithm()

print(f'Time: {time.time() - start_time}')

plot_front(nds['F'])

# =============================================================
# =============================================================

# from MOEAD import MOEAD
# from PMOP import MyPMOP
# from VisualizeFront import plot_front
# import time
#
# start_time = time.time()
# algorithm = MOEAD(n_gen=100, problem=MyPMOP(lambda_mass=0.0), pop_size=101, verbose=True)
#
# pop, nds = algorithm()
#
# print(f'Time: {time.time() - start_time}')
#
# plot_front(nds['F'])

# =============================================================
# =============================================================

