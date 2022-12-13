from typing import Dict, Any
from preprocessing.lib.dummy import DummyPipeline
from preprocessing.lib.categorical import OneHotColumnEncoder, CategoryEmbedder


class CategoryEncodingFactory:
    """
    Factory of category encoding methods
    """

    def __init__(self) -> None:
        self.encoding_methods = {
            "one_hot": OneHotColumnEncoder,
            "category_embedding": CategoryEmbedder,
            "passthrough": DummyPipeline,
        }
        self.default_params = {}

    def create_oversampler(
        self, method: str, params: Dict[str, Any] = None
    ) -> Any:
        if not params:
            params = self.default_params.get(method, {})
        return self.encoding_methods[method](**params)
