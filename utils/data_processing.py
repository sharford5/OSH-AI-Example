import pandas as pd
import numpy as np
import random

def load_data():
    df = pd.read_csv('./data/aw_fb_data.csv')
    return df


def process_data(df):
    #Convert Categorical
    # df['device'] = df['device'].astype('category')
    # df['device'] = df['device'].cat.codes
    # df['activity'] = df['activity'].astype('category')
    # df['activity'] = df['activity'].cat.codes

    columns = df.columns[1:-2]
    for c in columns:
        df[c] = (df[c] - df[c].min()) / (df[c].max() - df[c].min())

    return df


def split_data(df, p_test=20, p_val=10, random_seed=0):
    num_instances = df['Instance'].max()
    idx_list = [i for i in range(1,99)]
    random.Random(random_seed).shuffle(idx_list)

    p_train = num_instances - p_val - p_test

    train_idx = idx_list[:p_train]
    val_idx = idx_list[p_train:-p_test]
    test_idx = idx_list[-p_test:]

    columns = list(df.columns)[1:-1]

    df_train = df[df['Instance'].isin(train_idx)]
    df_val = df[df['Instance'].isin(val_idx)]
    df_test = df[df['Instance'].isin(test_idx)]

    X_train = df_train.iloc[:,:-1].values
    y_train = df_train.iloc[:,-1].values

    X_val = df_val.iloc[:,:-1].values
    y_val = df_val.iloc[:,-1].values

    X_test = df_test.iloc[:,:-1].values
    y_test = df_test.iloc[:,-1].values

    return X_train, X_val, X_test, y_train, y_val, y_test, columns