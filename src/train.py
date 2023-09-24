"""Functions to train model"""
# =================
# ==== IMPORTS ====
# =================

import os
import pandas as pd
import pickle

from comet_ml import Experiment
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor

from src.hyperparams import find_best_hyperparams
from src.logger import get_console_logger
from src.paths import MODELS_DIR
from src.preprocessing import pipeline_feature_engineering



# ===================
# ==== FUNCTIONS ====
# ===================

logger = get_console_logger()

def train(
    X: pd.DataFrame, y: pd.DataFrame,
    tune_hyperparams: bool=False, hyperparam_trials: int=10
) -> None:
    """
    """
    model_regressor = LGBMRegressor
    experiment = Experiment(
        api_key = os.environ["COMET_ML_API_KEY"],
        workspace=os.environ["COMET_ML_WORKSPACE"],
        project_name = "nyc-taxi-trip",
    )
    experiment.add_tag(model_regressor)

    # Split the data into train and test
    train_sample_size = int(0.8 * len(X))
    X_train, X_eval = X[:train_sample_size], X[train_sample_size:]
    y_train, y_eval = y[:train_sample_size], y[train_sample_size:]
    logger.info(f'Train sample size: {len(X_train)}')
    logger.info(f'Test sample size: {len(X_eval)}')

    if not tune_hyperparams:
        # Create the pipeline
        logger.info('Using the default parameters')
        pipeline = make_pipeline(
            pipeline_feature_engineering(),
            model_regressor()
        )
    else:
        logger.info('Finding best parameters using Bayesian optimization')
        best_preprocessing_hyperparams, best_model_hyperparams = find_best_hyperparams(
            model_regressor, hyperparam_trials, X_train, y_train, experiment
        )
        logger.info(f'Best preprocessing hyperparameters: {best_preprocessing_hyperparams}')
        logger.info(f'Best model hyperparameters: {best_model_hyperparams}')
        pipeline = make_pipeline(
            pipeline_feature_engineering(**best_preprocessing_hyperparams),
            model_regressor(**best_model_hyperparams)
        )
        experiment.add_tag('hyper-parameter-tuning')
    
    # train the model
    logger.info('Fitting model')
    pipeline.fit(X_train, y_train)

    # compute eval RMSE
    pred_eval = pipeline.predict(X_eval)
    rmse_eval = mean_squared_error(y_eval, pred_eval, squared=False)
    logger.info(f'Eval RMSE: {rmse_eval}')
    experiment.log_metrics({'rmse_eval': rmse_eval})

    # save the model to disk
    logger.info('Saving model to disk')
    with open(MODELS_DIR / 'model.pkl', "wb") as f:
        pickle.dump(pipeline, f)
    
    # log model artifact
    # experiment.log_model('eth-eur-1h-price-predictor', str(MODELS_DIR / 'model.pkl'))
    experiment.log_model(str(model_regressor), str(MODELS_DIR / 'model.pkl'))