import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor
from ProcessingData import normalization
from ReadExperiments import separate

a, b, c = separate()
df = normalization(a)


ann_model = MLPRegressor(max_iter=5000, activation='logistic', learning_rate='adaptive', n_iter_no_change=30,
                         hidden_layer_sizes=(256, 128), batch_size=400, learning_rate_init=0.01)

ann_h_model = ann_model.fit(X=np.array(df.drop(['Hcons', 'Pmech'], axis=1)), y=np.array(df['Hcons']))
ann_mech_model = ann_model.fit(X=np.array(df.drop(['Hcons', 'Pmech'], axis=1)), y=np.array(df['Pmech']))


joblib.dump(filename='ann_h_model.joblib', value=ann_h_model)
joblib.dump(filename='ann_mech_model.joblib', value=ann_mech_model)
