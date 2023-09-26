import itertools
import pandas as pd
import numpy as np
import datetime
import time

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "plotly_white"

from sksurv.functions import StepFunction
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.metrics import integrated_brier_score, brier_score

import itertools
    
    
    
def plot_feat_imp(cols, coef):
    """
    Plots the feature importances in a bar graph.
    
    :param cols: list, Names of the features.
    :param coef: array-like, Coefficients corresponding to the features.
    :return: DataFrame, A DataFrame containing the features and their importance.
             Figure, A Plotly figure object representing the bar graph.
    """
    feat_importance = pd.DataFrame({
        "feature": cols,
        "coef": coef
    })
    feat_importance["coef_abs"] = abs(feat_importance.coef)
    feat_importance.sort_values(by='coef_abs', ascending=True, inplace=True)

    fig = px.bar(feat_importance, x="coef", y="feature", height= 500, width= 600)
    
    fig.update_layout(
        dict(
            xaxis={'title' : 'Coefficient'}, 
            yaxis={'title' : 'Feature'}
        )
    )
    
    return feat_importance, fig
    
def get_bier_score(df, y_train, y_test, survs, times, col_target = "duration", with_benchmark=True):
    """
    Calculates the Brier score for the given survival predictions.
    
    :param df: DataFrame, The input dataset.
    :param y_train: array-like, The training labels.
    :param y_test: array-like, The testing labels.
    :param survs: array-like, Survival functions for predictions.
    :param times: array-like, The times at which to calculate the Brier score.
    :param col_target: str, The column name for the duration target variable. Default is "duration".
    :param with_benchmark: bool, Whether to include benchmark models. Default is True.
    :return: dict, A dictionary containing Brier scores for different models.
    """
    if with_benchmark:
    
        km_func = StepFunction(
            *kaplan_meier_estimator(df["censored"].astype(bool), df[col_target])
        )
        
        preds = {
            'estimator': np.row_stack([ fn(times) for fn in survs]),
            'random': 0.5 * np.ones((df.shape[0], times.shape[0])),
            'kaplan_meier': np.tile(km_func(times), (df.shape[0], 1))
        }
        
    else:
        preds = {'estimator': np.row_stack([ fn(times) for fn in survs])}
        
    scores = {}
    for k, v in preds.items():
        scores[k] = integrated_brier_score(y_train, y_test, v, times)
    
    return scores


def get_bier_curve(y_train, y_test, survs, times):
    """
    Calculates the Brier score curve over different times.
    
    :param y_train: array-like, The training labels.
    :param y_test: array-like, The testing labels.
    :param survs: array-like, Survival functions for predictions.
    :param times: array-like, The times at which to calculate the Brier score.
    :return: list, A list containing Brier scores at different times.
    """
    preds = {'estimator': np.row_stack([fn(times) for fn in survs])}

    scores = []
    for t in times:
        preds = [fn(t) for fn in survs]
        _, score = brier_score(y_train, y_test, preds, t)
        scores.append(score[0])

    return scores
    

def fit_score(estimator, Xy, train_index, test_index, cols, col_target):
    """
    Fits the model on training data and scores it on test data.
    
    :param estimator: object, The estimator object implementing ‘fit’ and ‘score’.
    :param Xy: DataFrame, The input dataset.
    :param train_index: array-like, Indices of training samples.
    :param test_index: array-like, Indices of testing samples.
    :param cols: list, Names of the features.
    :param col_target: str, The column name for the duration target variable.
    :return: object, The fitted estimator object.
             float, The score of the estimator on the test data.
    """
    Xy_train = Xy.loc[train_index]
    Xy_test = Xy.loc[test_index]

    y_train = np.array(
        list(zip(Xy_train.censored, Xy_train[col_target])),
        dtype=[('censored', '?'), (col_target, '<f8')])

    y_test = np.array(
        list(zip(Xy_test.censored, Xy_test[col_target])),
        dtype=[('censored', '?'), (col_target, '<f8')])

    estimator = estimator.fit(Xy_train[cols], y_train)

    score = estimator.score(Xy_test[cols], y_test)
    
    return estimator, score   


def cv_fit_score(df, cv, estimator_fn, cols, col_target, params, drop_zero = True, verbose = False):
    """
    Performs cross-validation, fitting the model on different folds, and returns the scores for each fold.
    
    :param df: DataFrame, The input dataset.
    :param cv: cross-validation generator, Determines the cross-validation splitting strategy.
    :param estimator_fn: function, A function that returns an estimator object.
    :param cols: list, Names of the features.
    :param col_target: str, The column name for the duration target variable.
    :param params: dict, Parameters to pass to the estimator.
    :param drop_zero: bool, Whether to drop rows where the target variable is zero. Default is True.
    :param verbose: bool, Whether to print messages while running. Default is False.
    :return: object, The estimator object.
             dict, A dictionary containing cross-validation scores.
    """
    Xy = df[cols+["censored", col_target]].dropna().reset_index(drop=True)
    
    if drop_zero:
        index_z = Xy[Xy[col_target]==0].index
        Xy = Xy.drop(index_z, axis=0).reset_index(drop=True)
    
    y = list(zip(Xy.censored, Xy[col_target]))
    y = np.array(y, dtype=[('censored', '?'), (col_target, '<f8')])
    
    cv_scores = {}

    t0 = time.time()
    for i, (train_index, test_index) in enumerate(cv.split(Xy)):

        estimator = estimator_fn(**params)
        estimator, score = fit_score(estimator, Xy, train_index, test_index, cols, col_target)

        if verbose:
            print(f"Fold {i}: {round(score, 3)}")

        cv_scores["fold_"+str(i)] = score
    
    
    cv_scores["time"] = (time.time() - t0)/60
    
    return estimator, cv_scores


def grid_search(grid_params, df, cv, estimator_fn, cols, col_target, verbose = False):
    """
    Performs grid search over a range of parameters to find the best estimator.
    
    :param grid_params: dict, Dictionary with parameters names (str) as keys and lists of parameter settings to try as values.
    :param df: DataFrame, The input dataset.
    :param cv: cross-validation generator, Determines the cross-validation splitting strategy.
    :param estimator_fn: function, A function that returns an estimator object.
    :param cols: list, Names of the features.
    :param col_target: str, The column name for the duration target variable.
    :param verbose: bool, Whether to print messages while running. Default is False.
    :return: object, The best estimator object after grid search.
             DataFrame, A DataFrame containing the results of the grid search.
    """
    best_score = -100
    
    n = 1
    for k, v in grid_params.items():
        n*=len(v)
        
    print(f'{n} total scenario to run')
    
    try: 
    
        for i, combi in enumerate(itertools.product(*grid_params.values())):
            params = {k:v for k,v in zip(grid_params.keys(), combi)}
            
            print(f'{i+1}/{n}: params: {params}')
            
            estimator, cv_scores = cv_fit_score(df, cv, estimator_fn, cols, col_target, params, verbose = verbose)
            
            table = pd.DataFrame.from_dict(cv_scores, orient='index').T
            cols_fold = [c for c in table.columns if 'fold' in c]
            table['mean'] = table[cols_fold].mean(axis=1)
            table['std'] = table[cols_fold].std(axis=1)
    
            for k, v in params.items():
                table[k] = v
    
            table = table[list(params.keys()) + [c for c in table.columns if c not in params]]
        
            results = table if i==0 else pd.concat([results, table], axis=0)
    
            if best_score < table['mean'].iloc[0]:
                best_score = table['mean'].iloc[0]
                best_estimator = estimator
    
    except KeyboardInterrupt:
        pass

    return best_estimator, results.reset_index(drop=True)