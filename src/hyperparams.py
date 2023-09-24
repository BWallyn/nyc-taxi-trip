"""Functions to find est hyperparameters"""
# =================
# ==== IMPORTS ====
# =================

from collections import defaultdict
import numpy as np
import pandas as pd
from typing import Callable, Union

from comet_ml import Experiment
from lightgbm import LGBMRegressor
import optuna
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from src.logger import get_console_logger
from sklearn.pipeline import make_pipeline
from src.preprocessing import get_preprocessing_pipeline

# Options
logger = get_console_logger()


# ===================
# ==== FUNCTIONS ====
# ===================

def sample_hyperparams(
    model_fn: Callable,
    trial: optuna.trial.Trial,
) -> dict[str, Union[str, int, float]]:

    if model_fn == Lasso:
        return {
            'alpha': trial.suggest_float('alpha', 0.01, 1.0, log=True)
        }
    elif model_fn == LGBMRegressor:
        return {
            "metric": 'mae',
            "verbose": -1,
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.2, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 3, 100),   
        }
    else:
        raise NotImplementedError('TODO: implement other models')


def find_best_hyperparams(
    model_regressor: Callable, hyperparam_trials: int,
    X: pd.DataFrame, y: pd.Series, experiment: Experiment,
) -> tuple[dict, dict]:
    """Find the best hyperparameters using bayesian optimization

    Args:
        model_regressor: Regressor model to optimize
        hyperparam_trials: Number of trials to find the best hyperparameters
        X: Dataset
        y: Target
        experiment: Experiment for Comet registry
    Returns:
    """
    assert model_regressor in {Lasso, LGBMRegressor}

    def objective(trial: optuna.trial.Trial) -> float:
        """Error function to minimize using hyperparameters tuning

        Args:
            trial: Optuna trial
        Returns:

        """
        # Sample hyperparameters
        model_hyperparams = sample_hyperparams(model_regressor, trial)

        # Use cross validation to analyze the performance of the model
        kf = KFold(n_splits=5, random_state=42)
        scores_train = []
        scores_eval = []
        for split_number, (train_ind, eval_ind) in enumerate(kf.split(X)):
            # Split data into train and evaluation
            X_train, X_eval = X.iloc[train_ind], X.iloc[eval_ind]
            y_train, y_eval = y.iloc[train_ind], y.iloc[eval_ind]

            # Log info
            logger.info(f'{split_number=}')
            logger.info(f'{len(X_train)=}')
            logger.info(f'{len(X_eval)=}')

            # Train the model
            pipeline = make_pipeline(
                get_preprocessing_pipeline(),
                model_regressor(**model_hyperparams)
            )
            pipeline.fit(X_train, y_train)

            # Evaluate the model
            pred_train = pipeline.predict(X_train)
            pred_eval = pipeline.predict(X_eval)
            rmse_train = mean_squared_error(y_train, pred_train, squared=False)
            rmse_eval = mean_squared_error(y_eval, pred_eval, squared=False)

            scores_train.append(rmse_train)
            scores_eval.append(rmse_eval)
            
            logger.info(f'{rmse_train=}')
            logger.info(f'{rmse_eval=}')

        score = np.array(scores_eval).mean()
        return score
    
    logger.info('Starting hyper-parameter search...')
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=hyperparam_trials)

    # Get the best hyperparameters and their values
    best_params = study.best_params
    best_value = study.best_value

    # Split best_params into preprocessing and model_hyperparameters
    best_preprocessing_hyperparams = {key: value for key, value in best_params.items() if key.startswith('pp_')}
    best_model_hyperparams = {key: value for key, value in best_params.items() if not key.startswith('pp_')}

    # Log
    logger.info("Best parameters:")
    for key, value in best_params.items():
        logger.info(f"{key}: {value}")
    logger.info(f"Best RMSE: {best_value}")

    experiment.log_metric('Cross_validation_RMSE', best_value)

    return best_preprocessing_hyperparams, best_model_hyperparams
