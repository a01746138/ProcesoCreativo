# Run the PMOEA algorithm
# ==========================================================
import numpy as np

from PMOEA import PMOEA
from PMOP import MyPMOP
import time

lambda_partitions = 10
algorithm = 'imia'
pop_size = 100
nuc = 1

if algorithm in ['nsga3', 'moead', 'imia']:
    n_gen = 600
elif algorithm == 'sms':
    n_gen = 60000
else:
    n_gen = 0

for i in range(1, 7):
    ex = (nuc - 1) * 6 + i
    if ex < 10:
        experiment = f'0{ex}'
    else:
        experiment = f'{ex}'

    start_time = time.time()
    run = PMOEA(n_gen=n_gen, pop_size=pop_size,
                lambda_partitions=lambda_partitions,
                problem=MyPMOP, algorithm=algorithm,
                verbose=True)

    run(experiment=experiment)
    simulation_time = time.time() - start_time
    np.savetxt(X=[simulation_time], fname=f'../MOEATimes/{algorithm}_exp{experiment}.txt')
