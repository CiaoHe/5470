from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import *
from sklearn.linear_model import *
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.neural_network import *
import xgboost as xgb
import catboost as cat
from sklearn.preprocessing import StandardScaler, scale
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split, KFold
import lightgbm as lgb
import sklearn
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline


train, test = pd.read_csv('train.csv'), pd.read_csv('test.csv')
data = pd.concat([train, test], ignore_index=True)
# some features are masked by 0, recover
data.loc[data.age == 0, 'age'] = None
data.loc[data.charge_sensitivity == 0,
         'charge_sensitivity'] = data.charge_sensitivity.median()
data.internet_age = data.internet_age.apply(lambda x: x/12)

trn_data, test_data = data[:len(train)], data[len(train):]
trn_data.drop(['id', 'age', ], axis=1, inplace=True)
del test_data['credit']
target = trn_data['credit']
del trn_data['credit']

X_trn, X_val, y_trn, y_val = train_test_split(trn_data, target, test_size=0.2)


def base_model(regressor, X_trn, X_val, y_trn, y_val, K_fold=5, Scale=False):
    X_trn, y_trn = X_trn.to_numpy(), y_trn.to_numpy()
    X_val, y_val = X_val.to_numpy(), y_val.to_numpy()
    scaler = StandardScaler()
    if Scale:
        pipe = Pipeline([('PreTransformer', scaler),
                         ('Regressor', regressor)])
    else:
        pipe = Pipeline([('Regressor', regressor)])
    print(pipe)

    y_pred = np.zeros_like(y_val)
    folds = KFold(n_splits=K_fold, shuffle=True, random_state=0)
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_trn, y_trn)):
        print(f"\n======fold No.{fold_+1}======")
        print('CV train_sample size : %d' % len(trn_idx))
        trn_X, trn_y = X_trn[trn_idx], y_trn[trn_idx]
        val_X, val_y = X_trn[val_idx], y_trn[val_idx]
        # fit
        pipe.fit(trn_X, trn_y)
        # predict on k_fold splitted val_X
        val_y_pred = pipe.predict(val_X)
        print('CV MAE Loss: %.2f' % mean_absolute_error(val_y_pred, val_y))
        # predict on X_val
        y_pred += pipe.predict(X_val)
    # div K-fold number
    y_pred /= K_fold
    # final metric on val-set
    MAE = mean_absolute_error(y_pred, y_val)
    print('\nMAE Loss: %.5f' % MAE)
    print('Final Score: %.5f' % (1/(1+MAE)))


def lgb_model(X_trn, X_val, y_trn, y_val, K_fold=10):
    X_trn, y_trn = X_trn.to_numpy(), y_trn.to_numpy()
    X_val, y_val = X_val.to_numpy(), y_val.to_numpy()
    y_pred = np.zeros_like(y_val)
    folds = KFold(n_splits=K_fold, shuffle=True, random_state=0)

    param = {
        'num_leaves': 27,
        'min_data_in_leaf': 20,
        'objective': 'regression_l1',
        'max_depth': 5,
        'learning_rate': 0.0081,
        "min_child_samples": 30,
        "boosting": "gbdt",
        "feature_fraction": 0.7,
        "bagging_freq": 1,
        "bagging_fraction": 0.8,
        "bagging_seed": 11,
        "metric": 'mae',
        "lambda_l1": 0.60,
        "verbosity": -1
    }

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_trn, y_trn)):
        print(f"\n======fold No.{fold_+1}======")
        print('CV train_sample size : %d' % len(trn_idx))
        trn_data = lgb.Dataset(X_trn[trn_idx], y_trn[trn_idx])
        val_data = lgb.Dataset(X_trn[val_idx], y_trn[val_idx])
        num_round = 10000
        regressor = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data],
                              verbose_eval=1000, early_stopping_rounds=500)
        y_pred += regressor.predict(X_val,
                                    num_iteration=regressor.best_iteration) / K_fold

    # final metric on val-set
    MAE = mean_absolute_error(y_pred, y_val)
    print('\nMAE Loss: %.5f' % MAE)
    print('Final Score: %.5f' % (1/(1+MAE)))


def xgb_model(X_trn, X_val, y_trn, y_val, K_fold=10):
    X_trn, y_trn = X_trn.to_numpy(), y_trn.to_numpy()
    X_val, y_val = X_val.to_numpy(), y_val.to_numpy()
    y_pred = np.zeros_like(y_val)
    folds = KFold(n_splits=K_fold, shuffle=True, random_state=0)

    xgb_params = {'eta': 0.004, 'max_depth': 6, 'subsample': 0.5, 'colsample_bytree': 0.5, 'alpha': 0.2,
                  'objective': 'reg:gamma', 'eval_metric': 'mae', 'silent': True, 'nthread': -1
                  }

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_trn, y_trn)):
        print(f"\n======fold No.{fold_+1}======")
        print('CV train_sample size : %d' % len(trn_idx))
        trn_data = xgb.DMatrix(X_trn[trn_idx], y_trn[trn_idx])
        val_data = xgb.DMatrix(X_trn[val_idx], y_trn[val_idx])

        watch_list = [(trn_data, 'train'), (val_data, 'valid_data')]
        regor = xgb.train(dtrain=trn_data, num_boost_round=10000, evals=watch_list, early_stopping_rounds=200,
                          verbose_eval=1000, params=xgb_params)
        y_pred += regor.predict(xgb.DMatrix(X_val),
                                ntree_limit=regor.best_ntree_limit) / K_fold

    # final metric on val-set
    MAE = mean_absolute_error(y_pred, y_val)
    print('\nMAE Loss: %.5f' % MAE)
    print('Final Score: %.5f' % (1/(1+MAE)))


def xgb_sklearn(X_trn, X_val, y_trn, y_val, K_fold=10):
    X_trn, y_trn = X_trn.to_numpy(), y_trn.to_numpy()
    X_val, y_val = X_val.to_numpy(), y_val.to_numpy()
    y_pred = np.zeros_like(y_val)
    folds = KFold(n_splits=K_fold, shuffle=True, random_state=0)

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_trn, y_trn)):
        print(f"\n======fold No.{fold_+1}======")
        print('CV train_sample size : %d' % len(trn_idx))
        trn_X, trn_y = X_trn[trn_idx], y_trn[trn_idx]
        val_X, val_y = X_trn[val_idx], y_trn[val_idx]
        regor = xgb.XGBRegressor(n_estimators=10000, max_depth=6, learning_rate=0.004, subsample=0.5,
                                 colsample_bytree=0.5, reg_alpha=0.2, n_jobs=-1, verbosity=1)

        regor.fit(trn_X, trn_y, eval_metric='mae', eval_set = [(val_X,val_y)],
                  early_stopping_rounds=200, verbose=True)
        y_pred += regor.predict(X_val,
                                ntree_limit=regor.best_ntree_limit) / K_fold

    # final metric on val-set
    MAE = mean_absolute_error(y_pred, y_val)
    print('\nMAE Loss: %.5f' % MAE)
    print('Final Score: %.5f' % (1/(1+MAE)))

    #save model
    regor.save_model('save_models/xgb.json')


cat_params_v1 = {'depth': 6, 'learning_rate': 0.8, 'l2_leaf_reg': 2, 'num_boost_round': 10000, 'random_seed': 94,
                 'loss_function': 'MAE'}

cat_params_v2 = {
    'n_estimators': 10000,
    'learning_rate': 0.02,
    'random_seed': 4590,
    'reg_lambda': 0.08,
    'subsample': 0.7,
    'bootstrap_type': 'Bernoulli',
    'boosting_type': 'Plain',
    'one_hot_max_size': 10,
    'rsm': 0.5,
    'leaf_estimation_iterations': 5,
    'use_best_model': True,
    'max_depth': 6,
    'verbose': -1,
    'thread_count': 4
}


def cat_model(X_trn, X_val, y_trn, y_val, K_fold=10, cat_params=cat_params_v1):
    X_trn, y_trn = X_trn.to_numpy(), y_trn.to_numpy()
    X_val, y_val = X_val.to_numpy(), y_val.to_numpy()
    y_pred = np.zeros_like(y_val)
    folds = KFold(n_splits=K_fold, shuffle=True, random_state=0)

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_trn, y_trn)):
        print(f"\n======fold No.{fold_+1}======")
        print('CV train_sample size : %d' % len(trn_idx))
        regor = cat.CatBoostRegressor(**cat_params)

        regor.fit(X_trn[trn_idx], y_trn[trn_idx], early_stopping_rounds=200, verbose_eval=1000,
                  use_best_model=True, eval_set=(X_trn[val_idx], y_trn[val_idx]))
        y_pred += regor.predict(X_val) / K_fold

    # final metric on val-set
    MAE = mean_absolute_error(y_pred, y_val)
    print('\nMAE Loss: %.5f' % MAE)
    print('Final Score: %.5f' % (1/(1+MAE)))


def cat_grid_search(X_trn, y_trn):
    X_trn, y_trn = X_trn.to_numpy(), y_trn.to_numpy()
    cat_params = {'num_boost_round': 10000, 'random_seed': 94,
                  'loss_function': 'MAE'}
    model = cat.CatBoostRegressor(**cat_params)
    grid = {'learning_rate': [0.03, 0.1],
            'depth': [4, 6, 10],
            'l2_leaf_reg': [1, 3, 5, 7, 9]}
    grid_search_result = model.grid_search(grid,
                                           X=X_trn,
                                           y=y_trn,
                                           verbose=1000,
                                           plot=False)
    print(grid_search_result)


xgb_sklearn(X_trn, X_val, y_trn, y_val, K_fold=10)
# cat_grid_search(X_trn, y_trn)
