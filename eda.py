import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
import streamlit as st

warnings.simplefilter("ignore")

PATH = 'https://raw.githubusercontent.com/nedokormysh/lin_models_presentation/main/datasets/'

DATASETS_NAMES = ['D_clients.csv',
                  'D_close_loan.csv',
                  'D_job.csv',
                  'D_last_credit.csv',
                  'D_loan.csv',
                  'D_pens.csv',
                  'D_salary.csv',
                  'D_target.csv',
                  'D_work.csv']

# datasets = !ls ../data/datasets
# datasets = [df.split('.')[0] for df in datasets]

# df_dict = {df: pd.read_csv("../data/datasets/" + df + ".csv") for df in datasets}
@st.cache_data
def open_data(path:str=PATH, datasets_names:list=DATASETS_NAMES):
    '''
    :param path: path to datasets
    :type path: str
    :param datasets_names: names of datasets
    :type datasets_names: list[str]
    :return: tuple[list[pandas.DataFame], dict[str, pandas.DataFrame], list[str]]

    ..notes::
    function to download files from github
    '''


    clients, close_loan, job, last_credit, loan, pens, salary, target, work = [pd.read_csv(path + f'{i}') for i in
                                                                               datasets_names]

    print('Datasets dowloaded')
    datasets = [clients, close_loan, job, last_credit, loan, pens, salary, target, work]

    df_dict = {}

    for i in range(len(DATASETS_NAMES)):
        name = re.sub(r'D_', '', DATASETS_NAMES[i].split('.')[0])
        df_dict[name] = datasets[i]

    return datasets, df_dict, datasets_names

@st.cache_data
def table_na_duble(datasets, datasets_name=DATASETS_NAMES):
    '''
    :param datasets:
    :type datasets: list[pandas.DataFame]
    :param datasets_names: names of datasets
    :type datasets_names: list[str]
    :return: datafame of parameters of datasets
    :rtype: pandas.DataFrame
    '''
    df_origin_table = pd.DataFrame(columns=['Название датасета',
                                            'Количество дублей',
                                            'Количество пропусков (сумма по всем колонкам)'])

    for i in range(len(datasets)):
        name = re.sub(r'D_', '', datasets_name[i].split('.')[0])
        empty = datasets[i].isna().any().any()
        if empty:
            empty = datasets[i].isna().sum().sum()

        duplicates = datasets[i].duplicated().sum()
        df_origin_table.loc[len(df_origin_table.index)] = name, duplicates, empty

    return df_origin_table

@st.cache_data
def data_processing(df_dict):
    '''
    :param df_dict: dict with source dataframes
    :type df_dict: dict[str, pandas.DataFrame]
    :return: merged dataframe of data in df_dict, list of continuous features, list of categorical features, list of some categorical features
    :rtype: tuple[pandas.DataFrame, list[str], list[str], list[str]]

    ..notes::
    function to merge datasets in single dataset
    '''

    # LOAN
    all_loan = df_dict['loan'].copy()
    all_loan = all_loan.merge(df_dict['close_loan'], on="ID_LOAN", how="inner")
    all_loan = all_loan.groupby("ID_CLIENT").agg({"ID_LOAN": "count", "CLOSED_FL": "sum"})
    all_loan.rename(columns={'ID_LOAN': 'LOAN_AMOUNT', 'CLOSED_FL': 'CLOSED_LOANS'}, inplace=True)
    # CLIENT
    all_data = df_dict['clients'].copy()
    all_data.rename(columns={'ID': 'ID_CLIENT'}, inplace=True)

    all_data = all_data.merge(df_dict['target'], on='ID_CLIENT', how='right')  # таргет.
    all_data = all_data.merge(df_dict['job'], on="ID_CLIENT", how="left")  # работа
    all_data = all_data.merge(df_dict['salary'], on="ID_CLIENT", how="left")  # зарплата
    all_data = all_data.merge(all_loan, on='ID_CLIENT', how='left')  # информация о кредитах
    all_data = all_data.merge(df_dict['last_credit'], on="ID_CLIENT", how="left")  # информация о последнем кредите

    all_data.drop(columns=['ID_CLIENT', 'AGREEMENT_RK'], inplace=True)

    continuous_features = ['CREDIT', 'FST_PAYMENT', 'AGE', 'CHILD_TOTAL', 'DEPENDANTS',
                           'OWN_AUTO', 'PERSONAL_INCOME',
                           # 'WORK_TIME',
                           'CLOSED_LOANS', 'LOAN_AMOUNT']

    categorical_features = [i for i in all_data.columns if i not in continuous_features and i != 'TARGET']

    cat_feats = ['GENDER', 'EDUCATION', 'MARITAL_STATUS', 'SOCSTATUS_WORK_FL',
                 'SOCSTATUS_PENS_FL', 'FL_PRESENCE_FL', 'GEN_INDUSTRY', 'GEN_TITLE',
                 'JOB_DIR', 'FAMILY_INCOME', 'TERM']

    all_data.drop_duplicates(inplace=True)

    all_data['WORK_TIME_IN_YEARS'] = all_data['WORK_TIME'] / 12
    all_data = all_data[all_data['WORK_TIME_IN_YEARS'] < all_data['AGE'].max()]

    continuous_features.append('WORK_TIME_IN_YEARS')

    all_data = all_data[all_data['AGE'] > all_data['WORK_TIME_IN_YEARS']]
    all_data = all_data[all_data['AGE'] - 14 >= all_data['WORK_TIME_IN_YEARS']]

    all_data = all_data[all_data['PERSONAL_INCOME'] > all_data['PERSONAL_INCOME'].quantile(0.001)]

    all_data['LOAN_NOW'] = 0
    all_data.loc[all_data['LOAN_AMOUNT'] - all_data['CLOSED_LOANS'] != 0, ['LOAN_NOW']] = 1

    categorical_features.append('LOAN_NOW')
    cat_feats.append('LOAN_NOW')

    for el in categorical_features:
        all_data[el] = all_data[el].astype('category')

    all_data = all_data[[#'ID_CLIENT',
                         'AGE', 'GENDER', 'EDUCATION', 'MARITAL_STATUS',
                         'CHILD_TOTAL', 'DEPENDANTS', 'SOCSTATUS_WORK_FL',
                         'SOCSTATUS_PENS_FL', 'REG_ADDRESS_PROVINCE',
                         'FACT_ADDRESS_PROVINCE', 'POSTAL_ADDRESS_PROVINCE',
                         'FL_PRESENCE_FL', 'OWN_AUTO',
                         #'AGREEMENT_RK',
                         'GEN_INDUSTRY', 'GEN_TITLE', 'JOB_DIR',
                         # 'WORK_TIME',
                         'FAMILY_INCOME', 'PERSONAL_INCOME', 'LOAN_AMOUNT',
                         'CLOSED_LOANS', 'CREDIT', 'TERM', 'FST_PAYMENT', 'WORK_TIME_IN_YEARS', 'LOAN_NOW', 'TARGET']]

    return all_data, continuous_features, categorical_features, cat_feats


def corr_matrix(all_data, width=6, height=6):
    fig = plt.figure(figsize=(width, height))
    # fig, ax = plt.subplots()
    # sns.heatmap(all_data.corr(numeric_only=True).round(2), ax=ax)
    sns.set_style("whitegrid")


    mask = np.triu(np.ones_like(all_data.corr(numeric_only=True), dtype=bool))

    # print(all_data.head())

    heatmap = sns.heatmap(all_data.corr(numeric_only=True).round(2),
                          annot=True,
                          square=True,
                          cmap="BrBG",
                          cbar_kws={"fraction": 0.01},
                          linewidth=1,
                          mask=mask,
    )

    heatmap.set_title("Тепловая карта корреляции Пирсона", fontdict={"fontsize": 10}, pad=5)

    return fig

def show_continuous_grafics(all_data, feat, width=5, height=2):
    fig = plt.figure(figsize=(width, height))

    his = sns.histplot(data=all_data,
                       x=all_data[feat],
                       legend=True,
                       hue=all_data['TARGET'],
                       kde=True)

    his.set_title(f"График распределия {feat}. С разделением относительно целевой переменной", fontdict={"fontsize": 12}, pad=5)

    return fig

def pie_categorical_grafics(all_data, feat, labels, title, width, height):
    fig = plt.figure(figsize=(width, height))
    palette_color = sns.color_palette('deep')

    all_data[feat].value_counts().plot.pie(autopct='%1.3f%%', colors=palette_color, shadow=True, labels=labels).set(title=title)
    return fig

def show_categorical_grafics(all_data, feat, width=5, height=2):
    fig = plt.figure(figsize=(width, height))

    his = sns.histplot(data=all_data,
                       x=all_data[feat],
                       legend=True,
                       hue=all_data['TARGET'],
                       # kde=True
                       )

    his.set_title(f"График распределия {feat}. С разделением относительно целевой переменной", fontdict={"fontsize": 8}, pad=4)
    return fig

def target_grafics(all_data, width, height):
    labels = ['0: отклика не было', '1: отклик был зарегистрирован', ]
    title = 'Распределение целевой переменной'
    fig = plt.figure(figsize=(width, height))
    palette_color = sns.color_palette('deep')

    all_data['TARGET'].value_counts().plot.pie(autopct='%1.3f%%', colors=palette_color, shadow=True, labels=labels).set(title=title)
    return fig