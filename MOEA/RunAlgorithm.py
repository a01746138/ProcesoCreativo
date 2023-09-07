# Run the PMOEA algorithm
# ==========================================================

from PMOEA import PMOEA
from PMOP import MyPMOP

lambda_partitions = 10
algorithm = 'nsga3'
pop_size = 100
nuc = 1

if algorithm in ['nsga3', 'moead']:
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

    run = PMOEA(n_gen=n_gen, pop_size=pop_size,
                lambda_partitions=lambda_partitions,
                problem=MyPMOP, algorithm=algorithm,
                verbose=True)

    run(experiment=experiment)
