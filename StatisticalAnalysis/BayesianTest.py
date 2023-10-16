import pandas as pd
import numpy as np
import bayesiantests as bt
import matplotlib.pyplot as plt
import seaborn as sn

cv_h = pd.read_csv('CV_Hcons.csv')
cv_mech = pd.read_csv('CV_Pmech.csv')
rope = 0.01


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
                names = (i[6:], j[6:])
                bt.correlated_ttest(x=x, rope=rope, names=names, verbose=True)
                samples = bt.correlated_ttest_MC(x=x, rope=rope, nsamples=40732)
                sn.kdeplot(samples, fill=True)
                # plot rope region
                plt.axvline(x=-rope, color='orange')
                plt.axvline(x=rope, color='orange')
                # add label
                if i[:5] == 'Hcons':
                    plt.title(r'$f_1$ surrogate model comparison')
                elif i[:5] == 'Pmech':
                    plt.title(r'$f_2$ surrogate model comparison')
                plt.xlabel(f'{names[0].upper()} vs. {names[1].upper()}')
                plt.savefig(fname=f'../Images/posterior_{i[:5]}_{names[0].upper()}_{names[1].upper()}.png')
                plt.show()


bayesian_comparison(cv_h)
print('----------------------------------------')
bayesian_comparison(cv_mech)

