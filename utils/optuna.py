from typing import Dict, Any
import optuna
import numpy as np


def get_best_trial_params_cat(
    study: optuna.Study, model_difference_threshold: float = 0.05
) -> Dict[str, Any]:
    """
    Returns best trial's params from optuna study.
    Selects trial with best score with difference
    lower than specified threshold. Applies to classification
    models

    Args:
        study (optuna.Study): optuna study
        model_difference_threshold (float, optional):
            threshold of score difference between samples. Defaults to 0.05.

    Returns:
        Dict[str, Any]: params of best trial
    """
    trials = study.trials
    trials_filtered = list(
        filter(
            lambda trial: trial.values[1] < model_difference_threshold, trials,
        ),
    )
    if len(trials_filtered) != 0:
        trials = trials_filtered
    metrics = [trial.values[0] for trial in trials]
    max_metric_index = np.argmax(metrics)
    return trials[max_metric_index].params
