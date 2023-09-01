# from SMSEMOA import SMSEMOA
# from PMOP import MyPMOP
# import time
#
# start_time = time.time()
# algorithm = SMSEMOA(n_gen=1000, problem=MyPMOP(lambda_mass=0.5), verbose=True)
#
# pop = algorithm()
#
# print(f'Time: {time.time() - start_time}')
#
# print(pop['F'])

# =============================================================
# =============================================================

from MOEAD import MOEAD
from PMOP import MyPMOP
import time

start_time = time.time()
algorithm = MOEAD(n_gen=100, problem=MyPMOP(lambda_mass=0.0), pop_size=101, verbose=True)

pop = algorithm()

print(f'Time: {time.time() - start_time}')
