# Wilcoxon test for the results of the IMIA
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ranksums
import StatisticalAnalysis.bayesiantests as bt


def wilcoxon(algorithm_list, lam):
    for i in algorithm_list:
        algorithm_list.remove(i)
        for j in algorithm_list:
            if i != j:
                df1 = pd.read_csv(f'HV/{i}_hv_lambda{lam}.csv')
                df2 = pd.read_csv(f'HV/{j}_hv_lambda{lam}.csv')
                p = ranksums(df1[df1.columns[-1]], df2[df2.columns[-1]]).pvalue
                print(f'p-value of {i} and {j} when lambda is {lam}: {p}')


def norm(df):
    value_min, value_max = min_max(df)
    for col in df.columns:
        df[col] = (df[col] - value_min) / (value_max - value_min)

    return df


def min_max(df):
    min_value = np.inf
    max_value = -np.inf
    for col in df.columns:
        if df[col].min() < min_value:
            min_value = df[col].min()
        if df[col].max() > max_value:
            max_value = df[col].max()

    return min_value, max_value


def bayesian(algorithm_list, lam):
    data = pd.DataFrame([])
    for i in algorithm_list:
        df = pd.read_csv(f'HV/{i}_hv_lambda{lam}.csv')
        data[i] = df[df.columns[-1]]
    data = norm(data)
    for i in algorithm_list:
        algorithm_list.remove(i)
        for j in algorithm_list:
            if i != j:
                x = np.array(data[i], data[j])
                bt.correlated_ttest(x=x, rope=0.01, names=(i, j), verbose=True)


for el in range(10):
    print(f'Lambda: {el}')
    # wilcoxon(['sms', 'imia', 'pimia'], el)
    bayesian(['sms', 'imia', 'pimia'], el)
    print('========================')
