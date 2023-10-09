import itertools
import pandas as pd
import numpy as np
import datetime
import time

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "plotly_white"

from sksurv.metrics import concordance_index_censored
from sksurv.metrics import cumulative_dynamic_auc

def compute_score(censored, target, prediction, sign):
    """
    Computes the concordance index for the given parameters.
    
    Parameters:
    censored (array-like): Array indicating whether an observation is censored.
    target (array-like): True survival times or time of censoring.
    prediction (array-like): Predicted risk scores.
    sign (int): Directional modifier for the predictions.
    
    Returns:
    float: The computed concordance index.
    """
    return concordance_index_censored(list(censored.astype(bool)), target, sign*prediction)[0]

def compute_score_model(preds_df, col_var, col_pred, col_target, sign, metric="cindex", times = None):
    """
    Computes the concordance index for each unique value in col_var and returns them as a DataFrame.
    
    Parameters:
    preds_df (DataFrame): DataFrame containing predictions and true values.
    col_var (str): The column name containing the variable values.
    col_pred (str): The column name containing the prediction scores.
    col_target (str): The column name containing the target values.
    sign (int): Directional modifier for the predictions.
    metric (str, optional): The metric to compute, default is "cindex".
    times (array-like, optional): Array-like of time points where the function evaluates the cumulative dynamic AUC, default is None.
    
    Returns:
    DataFrame: DataFrame containing computed scores for each unique value in col_var.
    """
    scores = {}
    for k in preds_df[col_var].unique():
        tmp = preds_df[preds_df[col_var] == k]
        scores[k] = compute_score(tmp.censored, tmp[col_target], tmp[col_pred], sign)
    
    scores = pd.DataFrame.from_dict(scores, orient='index').reset_index().rename(
        columns={'index':col_var, 0:col_pred})
    return scores

def get_distrib(data, col_var, name):
    """
    Computes the distribution of col_var and its percentage.
    
    Parameters:
    data (DataFrame): Input data.
    col_var (str): The column name of the variable of interest.
    name (str): The name to be assigned to the percentage column.
    
    Returns:
    DataFrame: A DataFrame with the distribution and percentage of col_var.
    """
    cols_x = [c for c in data.columns if c !=col_var]
    distrib = data.groupby(col_var,as_index=False)[cols_x[0]].count()
    distrib[f'perc_{name}'] = distrib[cols_x[0]]/data.shape[0]*100
    distrib.drop(cols_x[0], axis=1, inplace=True)
    return distrib
    
def plot_score(scores_df, col_var, models_name):
    """
    Creates a plotly bar plot of scores for different models.
    
    Parameters:
    scores_df (DataFrame): DataFrame containing scores for different models.
    col_var (str): The column name containing the variable of interest.
    models_name (list): List of model names.
    
    Returns:
    plotly.graph_objects.Figure: The generated plotly figure.
    """
    scores_graph = pd.melt(
        scores_df[[col_var]+models_name], 
        id_vars=[col_var], value_name='score', var_name='model')
    scores_graph[col_var] = scores_graph[col_var].astype(str)
    scores_graph['score_round'] = scores_graph.score.round(3).astype('str')
    
    fig = px.bar(
        scores_graph, x='model', y='score', 
        color = col_var, barmode='group',
        text = 'score_round',
        color_discrete_sequence = ['royalblue','lightgrey']
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        dict(
            title = "{} - {}% of positive classes".format(
                col_var.capitalize(), 
                round(scores_df[scores_df[col_var]==1]['perc_train'].iloc[0])
            ),
            xaxis={'title' : 'Model'}, 
            yaxis={'title' : 'Concordance index', 'range': [0,1]},
        )
    )
    return fig
