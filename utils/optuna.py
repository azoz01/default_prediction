from typing import List, Dict, Any
import optuna
import numpy as np


def get_best_trial_params_cat(
    study: optuna.Study, model_difference_threshold=0.05
) -> Dict[str, Any]:

    trials: List = list(
        filter(
            lambda trial: trial.values[1] < model_difference_threshold,
            study.trials,
        ),
    )
    metrics: List[float] = [trial.values[0] for trial in trials]
    max_metric_index: int = np.argmax(metrics)
    return trials[max_metric_index].params


def get_best_trial_params_reg(study) -> Dict[str, Any]:
    diffs: List[float] = list(map(lambda trial: trial.values[1], study.trials))
    trials_with_lowest_diff_idx: np.ndarray[int] = np.argsort(diffs)[
        : int(0.3 * len(diffs))
    ]
    trials_with_lowest_diff: np.ndarray = np.array(study.trials)
    trials_with_lowest_diff = trials_with_lowest_diff[
        trials_with_lowest_diff_idx
    ]
    mses: List[float] = list(
        map(lambda trial: trial.values[0], trials_with_lowest_diff)
    )
    trial_with_lowest_mse_idx: int = np.argmin(mses)
    best_trial: optuna.Trial = trials_with_lowest_diff[
        trial_with_lowest_mse_idx
    ]
    params: Dict[str, Any] = best_trial.params
    return params
