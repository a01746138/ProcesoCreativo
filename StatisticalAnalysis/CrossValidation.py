# Perform a cross validation study for each one of the ML techniques
# ==========================================================

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import cross_validate
from ExperimentsMATLAB.ProcessingData import normalization


def x_validate(model, data_in, flag):
    if flag:
        x_val = cross_validate(estimator=model, X=np.array(data_in.drop(['Hcons', 'Pmech'], axis=1)),
                               y=np.array(data_in['Hcons']), cv=10, n_jobs=-1,
                               scoring='neg_mean_squared_error', return_estimator=True)
    else:
        x_val = cross_validate(estimator=model, X=np.array(data_in.drop(['Hcons', 'Pmech'], axis=1)),
                               y=np.array(data_in['Pmech']), cv=10, n_jobs=-1,
                               scoring='neg_mean_squared_error', return_estimator=True)
    return x_val


def cv_table(models_set, labels, flag):
    results = []
    for m in models_set:
        r = x_validate(model=m, data_in=df, flag=flag)
        results.append(-r['test_score'])
    return pd.DataFrame(np.array(results).T, columns=labels)


data = pd.read_csv(filepath_or_buffer='../ExperimentsMATLAB/Data.csv')
df = normalization(data)

model_h_ann = joblib.load(filename='../Surrogate/model_h_ann.joblib',
                          mmap_mode='r')
model_mech_ann = joblib.load(filename='../Surrogate/model_mech_ann.joblib',
                             mmap_mode='r')
model_h_svr = joblib.load(filename='../Surrogate/model_h_svr.joblib',
                          mmap_mode='r')
model_mech_svr = joblib.load(filename='../Surrogate/model_mech_svr.joblib',
                             mmap_mode='r')
model_h_dtr = joblib.load(filename='../Surrogate/model_h_dtr.joblib',
                          mmap_mode='r')
model_mech_dtr = joblib.load(filename='../Surrogate/model_mech_dtr.joblib',
                             mmap_mode='r')
model_h_rfr = joblib.load(filename='../Surrogate/model_h_rfr.joblib',
                          mmap_mode='r')
model_mech_rfr = joblib.load(filename='../Surrogate/model_mech_rfr.joblib',
                             mmap_mode='r')

h_models = [model_h_ann, model_h_svr, model_h_dtr, model_h_rfr]
mech_models = [model_mech_ann, model_mech_svr, model_mech_dtr, model_mech_rfr]
h_models_lbl = ['Hcons_ann', 'Hcons_svr', 'Hcons_dtr', 'Hcons_rfr']
mech_models_lbl = ['Pmech_ann', 'Pmech_svr', 'Pmech_dtr', 'Pmech_rfr']

h_results = cv_table(models_set=h_models, labels=h_models_lbl, flag=True)
mech_results = cv_table(models_set=mech_models, labels=mech_models_lbl, flag=False)
for j in range(9):
    h_results_e = cv_table(models_set=h_models, labels=h_models_lbl, flag=True)
    mech_results_e = cv_table(models_set=mech_models, labels=mech_models_lbl, flag=False)
    h_results = pd.concat([h_results, h_results_e])
    mech_results = pd.concat([mech_results, mech_results_e])
    print(j)


h_results.to_csv(path_or_buf='CV_Hcons.csv', index=False)
mech_results.to_csv(path_or_buf='CV_Pmech.csv', index=False)

print(h_results)
print(mech_results)
