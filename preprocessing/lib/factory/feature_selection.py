from typing import Dict, Any
from preprocessing.lib.feature_selection.rfe import RfeAdapter
from preprocessing.lib.feature_selection import FRUFSAdapter
from preprocessing.lib.dummy import DummyPipeline


class FeatureSelectionFactory:
    """
    Factory of feature selection methods
    """

    def __init__(self) -> None:
        self.encoding_methods = {
            "frufs": FRUFSAdapter,
            "rfe": RfeAdapter,
            "passthrough": DummyPipeline,
        }
        self.default_params = {}

    def create_oversampler(
        self, method: str, params: Dict[str, Any] = None
    ) -> Any:
        if not params:
            params = self.default_params.get(method, {})
        return self.encoding_methods[method](**params)
