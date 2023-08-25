from SMSEMOA import SMSEMOA
from PMOP import MyPMOP
import time

algorithm = SMSEMOA(n_gen=2000, problem=MyPMOP(lambda_mass=0.0), verbose=True)

algorithm()
