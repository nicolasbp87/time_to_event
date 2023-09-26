import itertools
import pandas as pd
import numpy as np
import datetime
import json
import os


import torch # For building the networks 
import torchtuples as tt # Some useful functions
from torchtuples.callbacks import Callback
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ReduceLROnPlateau
from pycox.evaluation import EvalSurv

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "plotly_white"

import itertools

# 1. Concordance Class:
# This class serves as a callback to calculate and log the concordance index at the end of each epoch, which is a performance 
# metric for survival models.

class Concordance(tt.cb.MonitorMetrics):
    """
    This class is an extension of tt.cb.MonitorMetrics and is used for monitoring
    Concordance metrics at the end of each epoch during training.
    
    Attributes
    ----------
    x : torch.Tensor
        The input data tensor.
    durations : array-like
        The array containing the durations for each observation in the dataset.
    events : array-like
        The array containing the event occurrences for each observation in the dataset.
    per_epoch : int, optional, default=1
        The frequency with which the Concordance metric should be calculated.
    discrete : bool, optional, default=False
        Whether the survival predictions are discrete or not.
    verbose : bool, optional, default=True
        Whether to print Concordance metric at the end of each epoch.
    """

    def __init__(self, x, durations, events, per_epoch=1, discrete=False, verbose=True):
        """
        Initializes the Concordance class with input data, durations, events, and other optional parameters.
        
        Parameters
        ----------
        x : torch.Tensor
            The input data tensor.
        durations : array-like
            Array containing the durations for each observation in the dataset.
        events : array-like
            Array containing the event occurrences for each observation in the dataset.
        per_epoch : int, optional, default=1
            The frequency with which the Concordance metric should be calculated.
        discrete : bool, optional, default=False
            Whether the survival predictions are discrete or not.
        verbose : bool, optional, default=True
            Whether to print Concordance metric at the end of each epoch.
        """

        super().__init__(per_epoch)
        self.x = x
        self.durations = durations
        self.events = events
        self.verbose = verbose
        self.discrete = discrete
    
    def on_epoch_end(self):
        """
        This method is called at the end of each epoch and is used for calculating and
        possibly printing the Concordance metric based on the current model, input data, 
        durations, and events.
        """

        super().on_epoch_end()
        if self.epoch % self.per_epoch == 0:
            if not(self.discrete):
                _ = self.model.compute_baseline_hazards()
                surv = self.model.predict_surv_df(self.x)
            else:
                surv = self.model.interpolate(10).predict_surv_df(self.x)
                
            ev = EvalSurv(surv, self.durations, self.events)
            concordance = ev.concordance_td()
            self.append_score('concordance', concordance)
            
            if self.verbose:
                print('concordance:', round(concordance, 5))

# 2. score_model Function:
# Used to score a trained model using the concordance index.
def score_model(model, data, durations, events, discrete=False):
    """
    Calculates and returns the Concordance metric for the given model and data.
    
    Parameters
    ----------
    model : object
        The model object to be scored.
    data : torch.Tensor
        The input data tensor.
    durations : array-like
        Array containing the durations for each observation in the dataset.
    events : array-like
        Array containing the event occurrences for each observation in the dataset.
    discrete : bool, optional, default=False
        Whether the survival predictions are discrete or not.
    
    Returns
    -------
    float
        The calculated Concordance metric for the given model and data.
    """

    if not(discrete):
        surv = model.predict_surv_df(data)
    else:
        surv = model.interpolate(10).predict_surv_df(data)
    return EvalSurv(surv, durations, events, censor_surv='km').concordance_td()

# 3. train_deep_surv Function:
# Generic training loop for deep survival models, allowing to train a model with the given parameters and datasets, 
# and then it evaluates the model on training, validation and test subsets and prints out the training and falidation losses.

def train_deep_surv(train, val, test, model_obj, out_features,
                    n_nodes, n_layers, dropout , lr =0.01, 
                    batch_size = 16, epochs = 500, output_bias=False,  
                    tolerance=10, 
                    model_params = {}, discrete= False,
                    print_lr=True, print_logs=True, verbose = True):
    """
    Trains a deep survival model on the provided training data and validates it 
    on the provided validation data. Also evaluates the model on test data and 
    returns training logs, the trained model, and scores on different datasets.
    
    Parameters
    ----------
    train : tuple
        Tuple containing training data and labels.
    val : tuple
        Tuple containing validation data and labels.
    test : tuple
        Tuple containing test data and labels.
    model_obj : object
        The survival model object to be trained.
    out_features : int
        Number of output features in the model.
    n_nodes : int
        Number of nodes in each layer.
    n_layers : int
        Number of hidden layers.
    dropout : float
        The dropout rate.
    lr : float, optional, default=0.01
        Learning rate.
    batch_size : int, optional, default=16
        Batch size for training.
    epochs : int, optional, default=500
        Number of training epochs.
    output_bias : bool, optional, default=False
        Whether to include output bias in the model.
    tolerance : int, optional, default=10
        The tolerance for early stopping.
    model_params : dict, optional, default={}
        Additional model parameters.
    discrete : bool, optional, default=False
        Whether the survival predictions are discrete or not.
    print_lr : bool, optional, default=True
        Whether to print learning rate during training.
    print_logs : bool, optional, default=True
        Whether to print training logs.
    verbose : bool, optional, default=True
        Whether to print verbose logs.

    Returns
    -------
    DataFrame
        The training logs as a DataFrame.
    object
        The trained model object.
    dict
        A dictionary containing the Concordance metric scores on train, validation, and test datasets.
    """
    
    in_features = train[0].shape[1]
    num_nodes = [n_nodes]*(n_layers)
    batch_norm = True
    
    net = tt.practical.MLPVanilla(
        in_features, num_nodes, out_features, 
        batch_norm, dropout, output_bias=output_bias)

    opt = torch.optim.Adam
    model = model_obj(net, opt, **model_params)
    model.optimizer.set_lr(lr)

    callbacks = [
        tt.callbacks.EarlyStopping(patience=15),
        Concordance(val[0], val[1][0], val[1][1], per_epoch=5, discrete=discrete)
    ]

    log = model.fit(train[0], train[1], batch_size, epochs, callbacks, verbose,
                val_data=val, val_batch_size=batch_size)

    logs_df = log.to_pandas().reset_index().melt(
        id_vars="index", value_name="loss", var_name="dataset").reset_index()
    
    #print("Last lr", lr_scheduler.get_last_lr())
        
    if print_logs:
        fig = px.line(logs_df, y="loss", x="index", color="dataset", width=800, height = 400)
        fig.show()
        
    # scoring the model
    scores = {
        'train': score_model(model, train[0], train[1][0], train[1][1]),
        'val': score_model(model, val[0], val[1][0], val[1][1]),
        'test': score_model(model, test[0], test[1][0], test[1][1])
    }
        
    return logs_df, model, scores

# 4. grid_search_deep Function:
# Performs agrid search over a predefined parameter space and returns the best model and a Dataframe with the results 
# of the different configurations.
def grid_search_deep(train, val, test, out_features, grid_params, model_obj):
    """
    Performs grid search on the provided grid parameters and returns the best model 
    along with the results table.
    
    Parameters
    ----------
    train : tuple
        Tuple containing training data and labels.
    val : tuple
        Tuple containing validation data and labels.
    test : tuple
        Tuple containing test data and labels.
    out_features : int
        Number of output features in the model.
    grid_params : dict
        Dictionary containing grid parameters for grid search.
    model_obj : object
        The survival model object to be trained.
    
    Returns
    -------
    object
        The best model object obtained from grid search.
    DataFrame
        The results table as a DataFrame containing scores and parameters for each combination in the grid search.
    """
    best_score = -100
    
    n = 1
    for k, v in grid_params.items():
        n*=len(v)
        
    print(f'{n} total scenario to run')
    
    result = {}
    
    try: 
        for i, combi in enumerate(itertools.product(*grid_params.values())):
            params = {k:v for k,v in zip(grid_params.keys(), combi)}

            params_ = params.copy()
            if 'model_params' in params_.keys():
                params_['model_params'] = {k:v for k,v in params['model_params'].items() if k!='duration_index'}

            print(f'{i+1}/{n}: params: {params_}')

            logs_df, model, scores = train_deep_surv(train, val, test, model_obj,out_features,
                                      print_lr=False, print_logs=False, verbose = True, **params)

            result[i] = {}
            for k, v in params_.items():
                result[i][k] = v
            result[i]['lr'] = model.optimizer.param_groups[0]['lr']
            for k, score in scores.items():
                result[i]['score_'+k] = score

            score = scores['test']
            print('Current score: {} vs. best score: {}'.format(score, best_score))

            if best_score < score:
                best_score = score
                best_model = model
    
    except KeyboardInterrupt:
        pass
        
    table = pd.DataFrame.from_dict(result, orient='index')
    
    return best_model, table.sort_values(by="score_test", ascending=False).reset_index(drop=True)

# 5. load_model Function:
# Loads a previously trained model from disk, allowing for resuming training or performing evaluations without retraining the model.
def load_model(filename, path, model_obj, in_features, out_features, params):
    """
    Loads a pre-trained model from the specified path and filename, 
    and returns the loaded model object.
    
    Parameters
    ----------
    filename : str
        The name of the file from which the model is to be loaded.
    path : str
        The path of the directory containing the model file.
    model_obj : object
        The survival model object to be loaded.
    in_features : int
        Number of input features in the model.
    out_features : int
        Number of output features in the model.
    params : dict
        Dictionary containing parameters to initialize the model object.
    
    Returns
    -------
    object
        The loaded model object.
    """
    num_nodes = [int(params["n_nodes"])] * (int(params["n_layers"]))
    del params["n_nodes"]
    del params["n_layers"]

    if 'model_params' in params.keys():
        model_params = json.loads(params['model_params'].replace('\'', '\"'))
        del params['model_params']
        net = tt.practical.MLPVanilla(
            in_features=in_features, out_features=out_features, num_nodes=num_nodes, **params)
        model = model_obj(net, **model_params)
    else:
        net = tt.practical.MLPVanilla(
            in_features=in_features, out_features=out_features, num_nodes=num_nodes, **params)
        model = model_obj(net)
    model.load_net(os.path.join(path, filename))

    return model