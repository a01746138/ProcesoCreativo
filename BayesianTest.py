import pandas as pd
import numpy as np
import bayesiantests as bt
import matplotlib.pyplot as plt
import seaborn as sn

cv_h = pd.read_csv('CV_Hcons.csv')
cv_mech = pd.read_csv('CV_Pmech.csv')


def min_max(df):
    min_value = np.inf
    max_value = -np.inf
    for col in df.columns:
        if df[col].min() < min_value:
            min_value = df[col].min()
        if df[col].max() > max_value:
            max_value = df[col].max()

    return min_value, max_value


def norm(df):
    value_min, value_max = min_max(df)
    for col in df.columns:
        df[col] = (df[col] - value_min) / (value_max - value_min)

    return df


def bayesian_comparison(df):
    df = norm(df)
    lbl = np.array(df.columns.values)
    for i in df.columns:
        lbl = np.delete(lbl, 0)
        for j in lbl:
            if i != j:
                x = np.array(df[[i, j]])
                names = (i, j)
                left, within, right = bt.correlated_ttest(x=x, rope=0.01, names=names, verbose=True)


bayesian_comparison(cv_h)
print('----------------------------------------')
bayesian_comparison(cv_mech)
