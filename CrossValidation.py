# Perform a cross validation study for each one of the surrogate models
# in the /surrogate\\ directory
# ==========================================================

import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import cross_validate
from ProcessingData import normalization
from ReadExperiments import separate


def x_validate(model, data, flag):
    if flag:
        x_val = cross_validate(estimator=model, X=data.drop(['Hcons', 'Pmech'], axis=1),
                               y=data['Hcons'], cv=10, n_jobs=-1,
                               scoring='neg_mean_squared_error')
    else:
        x_val = cross_validate(estimator=model, X=data.drop(['Hcons', 'Pmech'], axis=1),
                               y=data['Pmech'], cv=10, n_jobs=-1,
                               scoring='neg_mean_squared_error')
    return x_val


def df_create(model_set, labels):
    i = 0
    results = pd.DataFrame()
    for m in model_set:
        if i % 2 == 0:
            r = x_validate(model=m, data=df, flag=True)
        else:
            r = x_validate(model=m, data=df, flag=False)
        results[labels[i]] = r['test_score']
        i += 1
    return results


path = 'C:\\Users\\luiz4\\PycharmProjects\\ProcesoCreativo\\surrogate\\'
a, b, c = separate()
df = normalization(a)

ann_h_model = joblib.load(filename=path + 'ann_h_model.joblib',
                          mmap_mode='r')
ann_mech_model = joblib.load(filename=path + 'ann_mech_model.joblib',
                             mmap_mode='r')
svr_h_model = joblib.load(filename=path + 'svr_h_model.joblib',
                          mmap_mode='r')
svr_mech_model = joblib.load(filename=path + 'svr_mech_model.joblib',
                             mmap_mode='r')
dtr_h_model = joblib.load(filename=path + 'dtr_h_model.joblib',
                          mmap_mode='r')
dtr_mech_model = joblib.load(filename=path + 'dtr_mech_model.joblib',
                             mmap_mode='r')
rfr_h_model = joblib.load(filename=path + 'rfr_h_model.joblib',
                          mmap_mode='r')
rfr_mech_model = joblib.load(filename=path + 'rfr_mech_model.joblib',
                             mmap_mode='r')

models_lbl = ['ann_h', 'ann_mech',
              'svr_h', 'svr_mech',
              'dtr_h', 'dtr_mech',
              'rfr_h', 'rfr_mech']
models = [ann_h_model, ann_mech_model,
          svr_h_model, svr_mech_model,
          dtr_h_model, dtr_mech_model,
          rfr_h_model, rfr_mech_model]

cv_results = np.array(df_create(model_set=models, labels=models_lbl), dtype='float')

np.savetxt(fname='CrossValidation.txt', X=cv_results)
print(cv_results)


