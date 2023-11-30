import pandas as pd
import matplotlib.pyplot as plt


def initial_df():
    df = pd.read_csv('../ExperimentsMATLAB/Data.csv')
    df = (df - df.min()) / (df.max() - df.min())
    df['Class'] = 0
    df = df.drop(['Hcons', 'Pmech'], axis=1)
    return df


def nds_df(algorithm: str):
    df = pd.read_csv(f'../MOEAAnalysis/NDS/{algorithm}_nds_lambda0.csv')
    for lam in range(1, 10):
        df = pd.concat([df, pd.read_csv(f'../MOEAAnalysis/NDS/{algorithm}_nds_lambda{lam}.csv')])
    df['Class'] = 0
    df = df.drop(['Hcons', 'Pmech'], axis=1)
    return df


pd.plotting.parallel_coordinates(initial_df(), class_column='Class')
plt.gca().legend_.remove()
plt.xlabel('Decision variables')
plt.ylabel('Normalized values')
plt.title('LHS dataset')
plt.savefig(fname=f'../Images/pc_lhs.png')
plt.show()


label_dict = {'sms': 'SMS-EMOA', 'moead': 'MOEA/D', 'nsga3': 'NSGA-III',
              'imia': 'IMIA', 'pimia': 'P-IMIA'}

# for alg in ['imia', 'pimia']:
#     pd.plotting.parallel_coordinates(nds_df(alg), class_column='Class')
#     plt.gca().legend_.remove()
#     plt.xlabel('Decision variables')
#     plt.ylabel('Normalized values')
#     plt.title(f'{label_dict[alg]}')
#     plt.savefig(fname=f'../Images/pc_{alg}.png')
#     plt.show()
