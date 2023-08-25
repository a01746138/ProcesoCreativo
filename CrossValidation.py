# Perform a cross validation study for each one of the ML techniques
# ==========================================================

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate
from ProcessingData import normalization
from ReadExperiments import separate


def x_validate(model, data, flag):
    if flag:
        x_val = cross_validate(estimator=model, X=np.array(data.drop(['Hcons', 'Pmech'], axis=1)),
                               y=np.array(data['Hcons']), cv=10, n_jobs=-1,
                               scoring='neg_mean_squared_error', return_estimator=True)
    else:
        x_val = cross_validate(estimator=model, X=np.array(data.drop(['Hcons', 'Pmech'], axis=1)),
                               y=np.array(data['Pmech']), cv=10, n_jobs=-1,
                               scoring='neg_mean_squared_error', return_estimator=True)
    return x_val


def cv_table(models_set, labels, flag):
    results = []
    for m in models_set:
        r = x_validate(model=m, data=df, flag=flag)
        results.append(-r['test_score'])
    return pd.DataFrame(np.array(results).T, columns=labels)


a, b, c = separate()
df = normalization(a)


ann_model = MLPRegressor(max_iter=5000, activation='logistic', learning_rate='adaptive', n_iter_no_change=30,
                         hidden_layer_sizes=(256, 128), batch_size=400, learning_rate_init=0.01)
svr_model = SVR(epsilon=0.05, tol=1e-4)
dtr_model = DecisionTreeRegressor(min_samples_split=3)
rfr_model = RandomForestRegressor(n_estimators=20, min_samples_split=3, n_jobs=-1)

models = [ann_model, svr_model, dtr_model, rfr_model]
h_models_lbl = ['Hcons_ann', 'Hcons_svr', 'Hcons_dtr', 'Hcons_rfr']
mech_models_lbl = ['Pmech_ann', 'Pmech_svr', 'Pmech_dtr', 'Pmech_rfr']

h_results = cv_table(models_set=models, labels=h_models_lbl, flag=True)
mech_results = cv_table(models_set=models, labels=mech_models_lbl, flag=False)

h_results.to_csv(path_or_buf='CV_Hcons.csv', index=False)
mech_results.to_csv(path_or_buf='CV_Pmech.csv', index=False)

print(h_results)
print(mech_results)

