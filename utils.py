import random
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly as py
import plotly.graph_objs as go
import xgboost as xgb
from plotly import tools

import lightgbm as lgb


def load_files(TRAIN_FILE, TEST_FILE):
    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)

    tr_te = train.append(test)
    tr_te['ix_train'] = np.concatenate(
        (np.ones(len(train)), np.zeros(len(test))))
    tr_te.reset_index(drop=True, inplace=True)

    print("DONE: data loading")

    return tr_te, len(train), len(test)


def time_vars_parser(df, var_to_parse='datetime'):
    ix_date = pd.DatetimeIndex(df[var_to_parse])

    df['date'] = ix_date.date
    df['day'] = ix_date.day
    df['month'] = ix_date.month
    df['year'] = ix_date.year
    df['hour'] = ix_date.hour
    df['dow'] = ix_date.dayofweek
    df['woy'] = ix_date.weekofyear

    print("DONE: date parsing")

    return df


def make_me_log(df, y_vars=['casual', 'registered', 'count']):
    y_vars_log = []

    for y_var in y_vars:
        df[y_var + '_log'] = np.log(df[y_var] + 1)
        y_vars_log.append(y_var + '_log')

    return df, y_vars_log


def create_chilled(df, chilled_col, temp_min, temp_max, atemp_min, atemp_max):
    df[chilled_col] = np.zeros((len(df)))
    df.loc[((df.temp > temp_min) & (df.temp < temp_max))
           & ((df.atemp > atemp_min) & (df.atemp < atemp_max)), chilled_col] = 1

    print("DONE: chilled variable created")

    return df


def create_warm_but_dry(df, dry_col, hum_min, hum_max, atemp_min, atemp_max):
    df[dry_col] = np.zeros((len(df)))
    df.loc[((df.humidity > hum_min) & (df.humidity < hum_max))
           & ((df.atemp > atemp_min) & (df.atemp < atemp_max)), dry_col] = 1

    print("DONE: dry weather variable created")

    return df


def replace_holiday(df, year, month, day, replace):
    if replace == 'to_holiday':
        df.loc[(df.month == month) & (df.day == day)
               & (df.year == year), 'holiday'] = 1
        df.loc[(df.month == month) & (df.day == day)
               & (df.year == year), 'workingday'] = 0
    elif replace == 'to_work':
        df.loc[(df.month == month) & (df.day == day)
               & (df.year == year), 'holiday'] = 0
        df.loc[(df.month == month) & (df.day == day)
               & (df.year == year), 'workingday'] = 1
    else:
        raise Exception("columns to replace not found")

    print("DONE: holidays fix")

    return df


def create_school_holiday(df, school_col, month, day, year):
    df.loc[(df.month == month) & (df.day == day)
           & (df.year == year), school_col] = 1
    return df


def is_sunday(df, sunday_col="sunday"):
    df[sunday_col] = np.zeros((len(df)))
    df.loc[df.dow == 6, sunday_col] = 1

    print("DONE: sunday variable")

    return df


def rmsle(y_pred, y_actual):
    err = np.log(y_pred + 1) - np.log(y_actual + 1)
    mean_error = np.square(err).mean()
    return np.sqrt(mean_error)


def train_val_split(df, day=15):
    train = df[df['day'] <= day]
    val = df[df['day'] > day]

    print("Validation Size = ", '{:.3f}'.format(len(val) / len(df)))

    return train, val


def rush_binner(df, to_bin, n_bin, name):
    binned = pd.cut(to_bin, n_bin, retbins=False)
    df[name] = np.zeros((len(df), 1))

    for hour in range(len(binned)):
        df.loc[(df.workingday == 1) & (df.hour == hour), name] = binned[hour]

    df = df.join(pd.get_dummies(df[name], prefix=name))
    df = df.drop([name, name + '_0.0'], axis=1)

    print("DONE: binning")

    return df


def xgb_tuner(tr, n_models, x_vars, verbose=False):

    folds = defaultdict(list)
    diz = defaultdict(list)

    folds[0] = [1, 2, 3, 4]
    folds[1] = [5, 6, 7, 8]
    folds[2] = [9, 10, 11, 12]
    folds[3] = [13, 14, 15, 16]
    folds[4] = [17, 18, 19]
    shift_reg = 1
    shift_ca = 1

    for i in range(n_models):
        params = {}
        params['booster'] = 'gbtree'
        params['objective'] = "reg:linear"
        params['eta'] = 0.01
        params['gamma'] = random.uniform(0, 0.5)
        params['alpha'] = random.uniform(0, 0.5)
        params['lambda'] = random.uniform(0, 1)
        params['min_child_weight'] = random.uniform(0, 100)
        params['colsample_bytree'] = random.uniform(0.5, 0.99)
        params['subsample'] = random.uniform(0.5, 0.99)
        params['max_depth'] = random.randint(6, 15)
        params['max_delta_step'] = random.uniform(0, 10)
        params['silent'] = 1
        params['random_state'] = 1001

        r_ix = 0
        ca_ix = 0
        predictions = np.zeros(len(tr))

        for fold_day in range(len(folds)):
            train_fold = tr[eval(" | ".join(
                ["(tr['{0}'] == {1})".format('day', cond) for cond in folds[fold_day]]))]
            val_fold = tr[eval(" & ".join(
                ["(tr['{0}'] != {1})".format('day', cond) for cond in folds[fold_day]]))]

            d_train_r = xgb.DMatrix(train_fold[x_vars].values, label=np.log(
                train_fold['registered'] + shift_reg))
            d_valid_r = xgb.DMatrix(val_fold[x_vars].values, label=np.log(
                val_fold['registered'] + shift_reg))
            watchlist_r = [(d_train_r, 'train'), (d_valid_r, 'eval')]

            d_train_c = xgb.DMatrix(train_fold[x_vars].values, label=np.log(
                train_fold['casual'] + shift_ca))
            d_valid_c = xgb.DMatrix(val_fold[x_vars].values, label=np.log(
                val_fold['casual'] + shift_ca))
            watchlist_c = [(d_train_c, 'train'), (d_valid_c, 'eval')]

            model_r = xgb.train(params, d_train_r, num_boost_round=10000,  evals=watchlist_r,
                                early_stopping_rounds=20, verbose_eval=False)

            model_ca = xgb.train(params, d_train_c, num_boost_round=10000,  evals=watchlist_c,
                                 early_stopping_rounds=20, verbose_eval=False)

            y_pred_r = np.exp(model_r.predict(d_valid_r)) - shift_reg
            y_pred_c = np.exp(model_ca.predict(d_valid_c)) - shift_ca

            y_pred_comb = np.round(y_pred_r + y_pred_c)
            y_pred_comb[y_pred_comb < 0] = 0

            predictions[eval(" & ".join(["(tr['{0}'] != {1})".format(
                'day', cond) for cond in folds[fold_day]]))] = y_pred_comb

            r_ix += int(model_r.best_ntree_limit)
            ca_ix += int(model_ca.best_ntree_limit)

        rmsle_i = rmsle(predictions, tr['count'])
        ca_ix = int(ca_ix / len(folds))
        r_ix = int(r_ix / len(folds))
        diz['params'].append(params)
        diz['rmsle'].append(rmsle_i)
        diz['ca_ix'].append(ca_ix)
        diz['r_ix'].append(r_ix)

        if verbose:
            print("Model {0}: {1}".format(i, rmsle_i),
                  '\n r_ix = {}'.format(r_ix), '\n ca_ix = {}'.format(ca_ix))
    return diz


def lgb_tuner(tr, n_models, x_vars, verbose=False):

    folds = defaultdict(list)
    diz = defaultdict(list)

    folds[0] = [1, 2, 3, 4]
    folds[1] = [5, 6, 7, 8]
    folds[2] = [9, 10, 11, 12]
    folds[3] = [13, 14, 15, 16]
    folds[4] = [17, 18, 19]
    shift_reg = 1
    shift_ca = 1

    for i in range(n_models):
        predictions = np.zeros(len(tr))
        r_ix = 0
        ca_ix = 0
        params = {'task': 'train',
                  'boosting_type': 'gbdt',
                  'objective': 'regression',
                  'metric': {'l2'},
                  'min_data_in_leaf': random.randint(1, 100),
                  'num_leaves': random.randint(1, 100),
                  'learning_rate': 0.01,
                  'feature_fraction':  random.uniform(.5, .99),
                  'bagging_fraction':  random.uniform(.5, .99),
                  'bagging_freq': random.randint(2, 10),
                  'verbose': 0,
                  'lambda_l1': random.uniform(.0, 1.99),
                  'lambda_l2': random.uniform(.0, 1.99)}

        for fold_day in range(len(folds)):
            train_fold = tr[eval(" | ".join(
                ["(tr['{0}'] == {1})".format('day', cond) for cond in folds[fold_day]]))]
            val_fold = tr[eval(" & ".join(
                ["(tr['{0}'] != {1})".format('day', cond) for cond in folds[fold_day]]))]

            lgb_train_r = lgb.Dataset(train_fold[x_vars], np.log(
                train_fold['registered'] + shift_reg))
            lgb_eval_r = lgb.Dataset(val_fold[x_vars], np.log(
                val_fold['registered'] + shift_reg))

            lgb_train_ca = lgb.Dataset(train_fold[x_vars], np.log(
                train_fold['casual'] + shift_reg))
            lgb_eval_ca = lgb.Dataset(
                val_fold[x_vars], np.log(val_fold['casual'] + shift_reg))

            model_r = lgb.train(params,
                                lgb_train_r,
                                num_boost_round=10000,
                                valid_sets=lgb_eval_r, verbose_eval=False,
                                early_stopping_rounds=25)

            model_c = lgb.train(params,
                                lgb_train_ca,
                                num_boost_round=10000,
                                valid_sets=lgb_eval_ca, verbose_eval=False,
                                early_stopping_rounds=25)

            y_pred_r = np.exp(model_r.predict(
                val_fold[x_vars], num_iteration=model_r.best_iteration)) - shift_reg
            y_pred_c = np.exp(model_c.predict(
                val_fold[x_vars], num_iteration=model_c.best_iteration)) - shift_ca

            y_pred_comb = np.round(y_pred_r + y_pred_c)
            y_pred_comb[y_pred_comb < 0] = 0

            predictions[eval(" & ".join(["(tr['{0}'] != {1})".format(
                'day', cond) for cond in folds[fold_day]]))] = y_pred_comb

            r_ix += int(model_r.best_iteration)
            ca_ix += int(model_c.best_iteration)

        rmsle_i = rmsle(predictions, tr['count'])
        ca_ix = int(ca_ix / len(folds))
        r_ix = int(r_ix / len(folds))
        diz['params'].append(params)
        diz['rmsle'].append(rmsle_i)
        diz['ca_ix'].append(ca_ix)
        diz['r_ix'].append(r_ix)

        if verbose:
            print("Model {0}: {1}".format(i, rmsle_i),
                  '\n r_ix = {}'.format(r_ix), '\n ca_ix = {}'.format(ca_ix))
    return diz


def xgb_predict_test(tr, te, x_vars, params, r_ix, ca_ix, shift_ca=1, shift_reg=1):

    d_train_r = xgb.DMatrix(
        tr[x_vars].values, label=np.log(tr['registered'] + shift_reg))
    d_train_c = xgb.DMatrix(
        tr[x_vars].values, label=np.log(tr['casual'] + shift_ca))
    d_test = xgb.DMatrix(te[x_vars].values)

    model_r = xgb.train(params, d_train_r, num_boost_round=r_ix)
    model_ca = xgb.train(params, d_train_c, num_boost_round=ca_ix)

    y_pred_cas = model_ca.predict(d_test)
    y_pred_cas = np.exp(y_pred_cas) - shift_ca
    y_pred_reg = model_r.predict(d_test)
    y_pred_reg = np.exp(y_pred_reg) - shift_reg

    y_pred_comb = np.round(y_pred_reg + y_pred_cas)
    print("DONE: Predicting on test with XGBoost")
    return y_pred_comb, model_ca, model_r


def lgb_predict_test(tr, te, x_vars, params, r_ix, ca_ix, shift_ca=1, shift_reg=1):

    lgb_train_r = lgb.Dataset(tr[x_vars], np.log(tr['registered'] + shift_reg))
    lgb_train_ca = lgb.Dataset(tr[x_vars], np.log(tr['casual'] + shift_reg))

    gbm_r = lgb.train(params,
                      lgb_train_r,
                      num_boost_round=r_ix,
                      verbose_eval=False)

    gbm_ca = lgb.train(params,
                       lgb_train_ca,
                       num_boost_round=ca_ix,
                       verbose_eval=False)

    y_pred_cas = gbm_ca.predict(te[x_vars], num_iteration=ca_ix)
    y_pred_cas = np.exp(y_pred_cas) - shift_ca
    y_pred_reg = gbm_r.predict(te[x_vars], num_iteration=r_ix)
    y_pred_reg = np.exp(y_pred_reg) - shift_reg

    y_pred_comb = np.round(y_pred_reg + y_pred_cas)

    print("DONE: Predicting on test with LGBM")
    return y_pred_comb, gbm_ca, gbm_r


def plot_varimp(imp):
    top_vars = [x[0] for x in imp]
    top_scores = [x[1] for x in imp]

    layout = go.Layout(
        title='Variable importance',
        xaxis=dict(
            title='score'
        ),
        yaxis=dict(
            title='variable'
        )
    )

    data = [go.Bar(
            x=top_scores[::-1],
            y=top_vars[::-1],
            orientation='h'
            )]

    fig = go.Figure(data=data, layout=layout)
    py.offline.iplot(fig)
