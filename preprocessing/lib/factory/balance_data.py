from typing import Dict, Any

from imblearn.over_sampling import RandomOverSampler
from preprocessing.lib.balance_data import SmotencWrapper
from preprocessing.lib.dummy import DummyPipeline


class OversampleFactory:
    def __init__(self) -> None:
        self.oversample_methods = {
            "oversample": RandomOverSampler,
            "smotenc": SmotencWrapper,
            "passthrough": DummyPipeline,
        }

        self.default_params = {"oversample": {"random_state": 42}}

    def create_oversampler(
        self, method: str, params: Dict[str, Any] = None
    ) -> Any:
        if not params:
            params = self.default_params.get(method, {})
        return self.oversample_methods[method](**params)
