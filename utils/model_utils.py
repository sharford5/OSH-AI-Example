import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

parameters_fit = {
    'cv':3,
    'Decision Tree':{
            # 'pca__n_components': [.1, .3, .5, .7, .9, 1.0],
            },
    'XGBoost':{
            # 'pca__n_components': [.1, .3, .5, .7, .9, 1.0],
            'reg__n_estimators': [100,500,750, 1000,1250, 1500],
            'reg__min_child_weight': [3,5,7],
            'reg__gamma': [0.9,1.5, 2.0],
            'reg__random_state' : [0], # Updated from 'seed'
            'reg__max_depth': [3,5, 7]
            },
    'Support Vector Machine': {
        # 'pca__n_components': [.1, .3, .5, .7, .9, 1.0],
        'reg__kernel': ['linear', 'poly', 'rbf'],
        'reg__tol': [1e-3, 1e-5, 2e-5],
        'reg__C': [0.5, 0.7, 1, 1.25,1.5]
    },
    'Random Forest':{
            # 'pca__n_components': [.1, .3, .5, .7, .9, 1.0],
            'reg__n_estimators': [100,500,750, 1000,1250, 1500],
            'reg__max_depth': [3,5, 7],
            'reg__min_samples_split':[2, 3, 4],
            'reg__random_state' : [0], # Updated from 'seed'
            }
}


param_grid = {
    'cv':3,
    'Decision Tree':{
            # 'pca__n_components': [.1, .3, .5, .7, .9, 1.0],
            },
    'XGBoost':{
            # 'pca__n_components': [.1, .3, .5, .7, .9, 1.0],
            'n_estimators': [100,500,750, 1000,1250, 1500],
            'min_child_weight': [3,5,7],
            'gamma': [0.9,1.5, 2.0],
            'random_state' : [0], # Updated from 'seed'
            'max_depth': [3,5, 7]
            },
    'Support Vector Machine': {
        # 'pca__n_components': [.1, .3, .5, .7, .9, 1.0],
        'kernel': ['linear', 'poly', 'rbf'],
        'tol': [1e-3, 1e-5, 2e-5],
        'C': [0.5, 0.7, 1, 1.25,1.5]
    },
    'Random Forest':{
            # 'pca__n_components': [.1, .3, .5, .7, .9, 1.0],
            'n_estimators': [100,500,750, 1000,1250, 1500],
            'max_depth': [3,5, 7],
            'min_samples_split':[2, 3, 4],
            'random_state' : [0], # Updated from 'seed'
            }
}
