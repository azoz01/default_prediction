from typing import Union, List, Dict, Any
import yaml
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.base import TransformerMixin
from api.data_model import ClientData, ClientDataList


class DefaultApiService:

    CONFIG_PATH: str = "api/api_config.yaml"

    def __init__(self):
        with open(self.CONFIG_PATH) as f:
            config: Dict[Any] = yaml.load(f, yaml.CLoader)

        self.model: TransformerMixin = self._load_model(
            model_path=config["model_path"]
        )
        self.pipelines: List[TransformerMixin] = self._load_pipelines(
            pipelines_paths=config["pipelines_paths"]
        )

    def _load_pipelines(self, pipelines_paths: List[str]):
        pipelines = []
        for path in pipelines_paths:
            with open(path, "rb") as f:
                pipelines.append(pkl.load(f))
        return pipelines

    def _load_model(self, model_path: str):
        with open(model_path, "rb") as f:
            return pkl.load(f)

    def extract_data_from_body(self, data: Union[ClientData, ClientDataList]):
        data_dict = data.dict()
        if "data" in data_dict:
            return data_dict["data"]
        return data_dict

    def convert_json_data_to_pandas(
        self, data: Union[List[Dict[str, Any]], Dict[str, Any]]
    ):
        if type(data) is not list:
            data = [data]
        converted_frame = pd.DataFrame(data, index=list(range(len(data))))
        return converted_frame

    def make_prediction(self, data: pd.DataFrame):
        preprocessed: pd.DataFrame = self._preprocess_data(data)
        proba: List[float] = self.model.predict_proba(preprocessed)
        print(proba)
        predictions: List[int] = np.argmax(proba, axis=1)
        output = {
            "probabilities": proba[:, 1].tolist(),
            "predictions": predictions.tolist(),
        }
        print(output)
        return output

    def _preprocess_data(self, data):
        out_data = data
        for pipeline in self.pipelines:
            out_data = pipeline.transform(out_data)
        return out_data
