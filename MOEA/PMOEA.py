from SMSEMOA import SMSEMOA
from MOEAD import MOEAD
from NSGA3 import NSGA3


class PMOEA:

    def __init__(self,
                 n_gen=100,
                 pop_size=100,
                 problem=None,
                 lambda_partitions=10,
                 algorithm='sms',
                 verbose=False):
        self.n_gen = n_gen
        self.pop_size = pop_size
        self.problem = problem
        self.lambda_partitions = lambda_partitions
        self.verbose = verbose

        if algorithm not in ['sms', 'moead', 'nsga3']:
            print('Defined algorithm not found.')
        else:
            self.algorithm = algorithm

    def _lambda_vector(self):
        v = []
        for i in range(self.lambda_partitions):
            val = i / (self.lambda_partitions - 1)
            v.append(val)

        return v

    def _run_algorithm(self, lam):

        algorithm = NSGA3(n_gen=self.n_gen, problem=self.problem(lambda_mass=lam),
                          pop_size=self.pop_size, verbose=self.verbose)

        pop, nds = algorithm()

        return pop

    def _do(self):
        lambda_pop = {}
        lambda_vec = self._lambda_vector()

        for index in range(self.lambda_partitions):
            lam = lambda_vec[index]
            pop = self._run_algorithm(lam)

            lambda_pop[f'lambda{index}'] = pop

        return lambda_pop

    def __call__(self):
        return self._do()
