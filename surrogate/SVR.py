import joblib
import numpy as np
from sklearn.svm import SVR
from ProcessingData import normalization
from ReadExperiments import separate

a, b, c = separate()
df = normalization(a)


svr_model = SVR(epsilon=0.01, tol=1e-5)

svr_h_model = svr_model.fit(X=np.array(df.drop(['Hcons', 'Pmech'], axis=1)), y=np.array(df['Hcons']))
svr_mech_model = svr_model.fit(X=np.array(df.drop(['Hcons', 'Pmech'], axis=1)), y=np.array(df['Pmech']))


joblib.dump(filename='svr_h_model.joblib', value=svr_h_model)
joblib.dump(filename='svr_mech_model.joblib', value=svr_mech_model)
