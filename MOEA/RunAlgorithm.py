# Run the PMOEA algorithm
# ==========================================================

from PMOEA import PMOEA
from PMOP import MyPMOP

n_gen = 10
pop_size = 50
lambda_partitions = 2
algorithm = 'nsga3'
nuc = 1

for i in range(1, 7):
    ex = (nuc - 1) * 6 + i
    if ex < 10:
        experiment = f'0{ex}'
    else:
        experiment = f'{ex}'

    run = PMOEA(n_gen=n_gen, pop_size=pop_size,
                lambda_partitions=lambda_partitions,
                problem=MyPMOP, algorithm=algorithm,
                verbose=True)

    run(experiment=experiment)
