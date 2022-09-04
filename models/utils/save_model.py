import os
import json
import pickle as pkl
import utils.paths as paths


def save_model(model, name, get_params_func=None):
    with open(
        os.path.join(paths.MODELS_SERIALIZED_PATH, f"{name}.pkl"), "wb"
    ) as f:
        pkl.dump(model, f)
    with open(
        os.path.join(paths.MODELS_PARAMETERS_PATH, f"{name}.json"), "w"
    ) as f:
        if get_params_func is not None:
            params = get_params_func(model)
        else:
            params = model.get_params()
        json.dump(params, f, indent=4)

