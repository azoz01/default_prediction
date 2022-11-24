from typing import Callable
import logging
import pandas as pd
from sklearn.base import TransformerMixin
from utils import load_data, save_data, save_pickle

logger = logging.getLogger(__name__)


class GenericPipeline:
    """
    Generic pipeline used for executables.
    Useful when pipeline follows pattern:
        1. Read data
        2. Fit Transformer
        3. Transform Data
        4. Save data
        5. Save fitted transformer
        6. Validate data if validating function provided
    """

    def __init__(
        self,
        transformer: TransformerMixin,
        input_path: str,
        output_path: str,
        serialized_transformer_output_path: str,
        validate_data: Callable[[pd.DataFrame, pd.DataFrame], None] = None,
    ) -> None:
        self.transformer = transformer
        self.input_path = input_path
        self.output_path = output_path
        self.serialized_transformer_output_path = (
            serialized_transformer_output_path
        )
        self.validate_data = validate_data

    def run_pipeline(self) -> None:
        """
        Runs pipeline in steps:
            1. Read data
            2. Fit Transformer
            3. Transform Data
            4. Save data
            5. Save fitted transformer
        """

        X_train, y_train, X_valid, y_valid, X_test, y_test = load_data(
            path=self.input_path
        )
        logger.info("Fitting transformer")
        self.transformer.fit(X=X_train, y=y_train)
        logger.info("Transforming train data")
        X_train, y_train = self.transformer.transform(X=X_train, y=y_train)
        logger.info("Transforming validation data")
        X_valid, y_valid = self.transformer.transform(X=X_valid, y=y_valid)
        logger.info("Transforming test data")
        X_test, y_test = self.transformer.transform(X=X_test, y=y_test)

        if self.validate_data:
            logger.info("Validating output data")
            self.validate_data(X_train, y_train)
            self.validate_data(X_valid, y_valid)
            self.validate_data(X_test, y_test)

        save_data(
            path=self.output_path,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            X_test=X_test,
            y_test=y_test,
        )
        save_pickle(
            object=self.transformer,
            output_path=self.serialized_transformer_output_path,
        )
