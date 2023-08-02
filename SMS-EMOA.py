import numpy as np
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.operators.sampling.rnd import FloatRandomSampling  # Sampling
from pymoo.operators.selection.tournament import TournamentSelection  # Selection
from pymoo.operators.crossover.sbx import SBX  # Crossover
from pymoo.operators.mutation.pm import PM  # Mutation
from pymoo.algorithms.moo.sms import cv_and_dom_tournament
from pymoo.algorithms.moo.sms import LeastHypervolumeContributionSurvival  # Sorting
from PMOP import MyPMOP


n_slice = 1  # Number of slices of lambda: n_slices + 1 - Should be 10 or 11
n_exp = 1  # Number of experiments - Should be 31 experiments
n_pop = 300  # Number of individuals per population - Should be 300 ind
n_gen = 10  # Number of generations - Should be around 3,000 generations


algorithm = SMSEMOA(pop_size=n_pop,
                    sampling=FloatRandomSampling(),
                    selection=TournamentSelection(func_comp=cv_and_dom_tournament),
                    crossover=SBX(prob_exch=0.5),
                    mutation=PM(),
                    survival=LeastHypervolumeContributionSurvival()
                    )


for i in range(n_slice + 1):
    lam = float(i/n_slice)
    for j in range(n_exp):
        res = minimize(problem=MyPMOP(lambda_mass=lam),
                       algorithm=algorithm,
                       termination=('n_gen', n_gen),
                       seed=np.random.randint(0, 1e5),
                       verbose=True)
        front = []
        for k in range(len(res.F)):
            ind = []
            for m in range(5):
                ind.append(res.X[k][m])
            ind.append(lam)
            for m in range(2):
                ind.append(res.F[k][m])
            front.append(ind)
        np.savetxt(fname=f'runs\\sms_lam{lam}_ngen{n_gen}_exp{j}.txt',
                   X=np.array(front),
                   delimiter=',')


# plot = Scatter()
# plot.add(MyPMOP().pareto_front(), plot_type="line", color="black", alpha=0.7)
# plot.add(res.F, color="red")
# plot.show()
