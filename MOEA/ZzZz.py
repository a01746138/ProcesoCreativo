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

# from NSGA3 import NSGA3
# from PMOP import MyPMOP
# from VisualizeFront import plot_front
# import time
#
# start_time = time.time()
#
# n_gen = 400
# algorithm = NSGA3(n_gen=n_gen, problem=MyPMOP(lambda_mass=0.0),
#                   pop_size=100, verbose=True)
#
# pop, nds = algorithm()
#
# print(f'Time to run the algorithm for {n_gen} generations: {time.time() - start_time}')
#
# plot_front(nds['F'])
