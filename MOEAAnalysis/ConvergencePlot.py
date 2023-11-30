import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt


def save(algorithms: list):
    for lam in range(10):
        for a in algorithms:
            hv_list = []
            for i in range(1, 31):
                if i < 10:
                    exp = f'0{i}'
                else:
                    exp = f'{i}'
                history = np.loadtxt(fname=f'../MOEARuns/{a}_hv_lambda{lam}_exp{exp}.txt', delimiter=',')[:, 1]
                hv_list.append(list(history))
            columns = []
            if a != 'pimia':
                for j in range(120):
                    columns.append((j + 1) * 500)
            else:
                for j in range(100):
                    columns.append(j * 600 + 580)
            df = pd.DataFrame(np.array(hv_list), columns=columns)
            df.to_csv(path_or_buf=f'HV/{a}_hv_lambda{lam}.csv', index=False)


def plot():
    for lam in range(10):
        ax = plt.figure(dpi=300).add_subplot()
        data_sms = pd.read_csv(filepath_or_buffer=f'HV/sms_hv_lambda{lam}.csv')
        data_moead = pd.read_csv(filepath_or_buffer=f'HV/moead_hv_lambda{lam}.csv')
        data_nsga3 = pd.read_csv(filepath_or_buffer=f'HV/nsga3_hv_lambda{lam}.csv')
        n_gen = [int(x) for x in data_sms.columns]
        data_sms_m = data_sms.median()
        data_moead_m = data_moead.median()
        data_nsga3_m = data_nsga3.median()
        sb.lineplot(y=data_sms_m, x=n_gen, ax=ax)
        sb.lineplot(y=data_moead_m, x=n_gen, ax=ax, linestyle='dashed')
        sb.lineplot(y=data_nsga3_m, x=n_gen, ax=ax, linestyle='dotted')
        plt.xscale('log')
        plt.xlabel('Number of function evaluations', fontsize=20)
        plt.ylabel('Hypervolume value', fontsize=20)
        plt.tight_layout()
        plt.savefig(fname=f'../Images/convergence_lambda{lam}.png')


def plot2():
    for lam in range(10):
        ax = plt.figure(dpi=300).add_subplot()
        data_sms = pd.read_csv(filepath_or_buffer=f'HV/sms_hv_lambda{lam}.csv')
        data_imia = pd.read_csv(filepath_or_buffer=f'HV/imia_hv_lambda{lam}.csv')
        n_gen = [int(x) for x in data_sms.columns]
        data_sms_m = data_sms.median()
        data_imia_m = data_imia.median()
        sb.lineplot(y=data_sms_m, x=n_gen, ax=ax)
        sb.lineplot(y=data_imia_m, x=n_gen, ax=ax, linestyle='dashed')
        plt.xscale('log')
        plt.xlabel('Number of function evaluations', fontsize=20)
        plt.ylabel('Hypervolume value', fontsize=20)
        plt.tight_layout()
        plt.savefig(fname=f'../Images/convergence2_lambda{lam}.png')


def plot3():
    for lam in range(10):
        ax = plt.figure(dpi=300).add_subplot()
        data_imia = pd.read_csv(filepath_or_buffer=f'HV/imia_hv_lambda{lam}.csv')
        data_pimia = pd.read_csv(filepath_or_buffer=f'HV/pimia_hv_lambda{lam}.csv')
        data_sms = pd.read_csv(filepath_or_buffer=f'HV/sms_hv_lambda{lam}.csv')
        n_gen_pimia = [int(x) for x in data_pimia.columns]
        n_gen_imia = [int(x) for x in data_imia.columns]
        data_sms_m = data_sms.median()
        data_imia_m = data_imia.median()
        data_pimia_m = data_pimia.median()
        sb.lineplot(y=data_sms_m, x=n_gen_imia, ax=ax)
        sb.lineplot(y=data_imia_m, x=n_gen_imia, ax=ax, linestyle='dashed')
        sb.lineplot(y=data_pimia_m, x=n_gen_pimia, ax=ax, linestyle='dotted')
        plt.xscale('log')
        plt.xlabel('Number of function evaluations', fontsize=20)
        plt.ylabel('Hypervolume value', fontsize=20)
        plt.tight_layout()
        plt.savefig(fname=f'../Images/convergence3_lambda{lam}.png')


plot2()
plot3()
