from RunAlgorithm import RunAlgorithm


lam = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
n_exp = 1000

alg = RunAlgorithm(lam=lam, n_exp=n_exp)
alg.sms_evaluate()
