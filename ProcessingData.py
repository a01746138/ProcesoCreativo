# Normalize the data per column and divide the data
# into training and testing subsets
# ==========================================================

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from ReadExperiments import separate


def normalization(data):
    scaler = MinMaxScaler()
    data_norm = pd.DataFrame(scaler.fit_transform(data))
    return data_norm


def read_file(train_sz):
    # Read the file where data is located
    df, err, over = separate()

    # Normalize the data [0,1]
    df_norm = normalization(df)

    # Extract the X and y values from data set
    x = df_norm.drop(columns=[6, 7])
    y_h = df_norm[6]
    y_mech = df_norm[7]

    # Create the train and test partitions for fc and motor
    x_h_train, x_h_test, y_h_train, y_h_test = train_test_split(x, y_h,
                                                                train_size=train_sz,
                                                                random_state=1)
    x_mech_train, x_mech_test, y_mech_train, y_mech_test = train_test_split(x, y_mech,
                                                                            train_size=train_sz,
                                                                            random_state=1)
    # Join train and test subsets, respectively
    data_h_train = pd.concat([x_h_train, y_h_train], axis=1, join='inner', ignore_index=True)
    data_h_test = pd.concat([x_h_test, y_h_test], axis=1, join='inner', ignore_index=True)
    data_mech_train = pd.concat([x_mech_train, y_mech_train], axis=1, join='inner', ignore_index=True)
    data_mech_test = pd.concat([x_mech_test, y_mech_test], axis=1, join='inner', ignore_index=True)

    return data_h_train, data_h_test, data_mech_train, data_mech_test
