import logging
import torch
import torch.nn as nn
from typing import Any, Dict, List, Tuple, Union
from sklearn.base import TransformerMixin
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from fastai.tabular.all import *

logger = logging.getLogger(__name__)


class CategoryEmbedder(TransformerMixin):
    """
    Encodes specified categorical columns using category embedding technique.
    """

    def __init__(self, embedder_n_epochs: int, embedder_num_layers: int):
        self.embedder_n_epochs: int = embedder_n_epochs
        self.embedder_num_layers: int = embedder_num_layers

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs):
        """
        Fits pipeline

        Args:
            X (pd.DataFrame): X to fit
            y (pd.DataFrame): y to fit
        """
        X = self._categorize_categorical_variables(X)
        X, y = self._resample_data(X, y)
        tabular_pandas: TabularPandas = self._get_tabular_pandas(
            X=X, y=y, target_name="TARGET"
        )
        self.nn_model: Learner = self._prepare_nn_model(
            tabular_pandas=tabular_pandas
        )
        self._fit_nn_model(
            nn_model=self.nn_model, embedder_n_epochs=self.embedder_n_epochs
        )
        return self

    def transform(
        self, X: pd.DataFrame, y: pd.DataFrame = None, **kwargs
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Encodes specified categorical columns using category embedding technique.

        Args:
            X (pd.DataFrame): X to transform
            y (pd.DataFrame, optional): pd.DataFrame which is passed through.

        Returns:
            Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]: Data with 
                transformed values. If y is not None, then passed through.
        """
        X = self._categorize_categorical_variables(X)
        tabular_pandas: TabularPandas = self._get_tabular_pandas(
            X, valid_sample_frac=0
        )
        X_embedded: pd.DataFrame = self._embed_features(
            self.nn_model, tabular_pandas
        )

        if y is not None:
            return X_embedded, y
        return X_embedded

    def _categorize_categorical_variables(
        self, X: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Converts columns with non-numerical data types to "category" type.

        Args:
            X (pd.DataFrame): X to process

        Returns:
            pd.DataFrame: X with converted columns
        """
        logger.info("Categorizing non-numerical columns inside frame")
        X = X.copy()
        X[X.select_dtypes(exclude=["float64"]).columns.values] = X[
            X.select_dtypes(exclude=["float64"]).columns.values
        ].astype("category")
        return X

    def _resample_data(
        self, X: pd.DataFrame, y: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Makes data perfectly balanced by oversampling minority class

        Args:
            X (pd.DataFrame): X to resample
            y (pd.DataFrame): y to resample

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Resampled data
        """
        logger.info("Resampling data")
        ros: RandomOverSampler = RandomOverSampler()
        X_resampled, y_resampled = ros.fit_resample(X, y)
        return X_resampled, y_resampled

    def _get_tabular_pandas(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame = None,
        target_name: str = None,
        valid_sample_frac: float = 0.3,
    ) -> TabularPandas:
        """
        Converts pandas pd.DataFrame to TabularPandas from fastai. If y
        is passed, then X and y are concatenated and y is treated
        as target

        Args:
            X (pd.DataFrame): X to convert
            y (pd.DataFrame, optional): y to concatenate and convert. 
                Defaults to None.
            target_name (str, optional): Name of target variable. Defaults to None.
            valid_sample_frac (float, optional): Fraction, how much data
                is used as validation sample. Defaults to 0.3.

        Returns:
            TabularPandas: converted frames to TabularPandas
        """
        if y is not None:
            df: pd.DataFrame = pd.concat([X, y], axis=1)
        else:
            df: pd.DataFrame = X

        categorical_columns: List[str] = X.select_dtypes(
            exclude=["float64"]
        ).columns.values.tolist()
        continuous_columns: List[str] = X.select_dtypes(
            include=["float64"]
        ).columns.values.tolist()

        tabular_pandas: TabularPandas = TabularDataLoaders.from_df(
            df,
            procs=[Categorify],
            cat_names=categorical_columns,
            cont_names=continuous_columns,
            y_names=target_name,
            batchsize=2048,
            drop_last=False,
            valid_idx=df.sample(frac=valid_sample_frac, random_state=42).index,
        )

        return tabular_pandas

    def _prepare_nn_model(self, tabular_pandas: TabularPandas) -> Learner:
        """
        Prepares Learner object using passed tabular_pandas.
        Underlying neural network is prepared using heuristic.

        Args:
            tabular_pandas (TabularPandas): object used during
                creation of Learner

        Returns:
            Learner: object ready for fit
        """
        num_embeddings: int = sum(n for _, n in get_emb_sz(tabular_pandas))
        num_classes: int = tabular_pandas.ys.nunique().values[0]
        continuous_columns = tabular_pandas.cont_names
        layers = self._get_default_nn_layers(
            num_embeddings,
            num_continuous=len(continuous_columns),
            num_outputs=num_classes,
            num_layers=self.embedder_num_layers,
        )
        config: Dict[str, Any] = tabular_config(
            ps=[0.001] + (self.embedder_num_layers - 1) * [0.01], embed_p=0.04
        )

        nn_model = tabular_learner(
            dls=tabular_pandas,
            layers=layers,
            config=config,
            loss_func=CrossEntropyLossFlat(),
            metrics=accuracy,
            n_out=num_classes,
        )

        return nn_model

    def _get_default_nn_layers(
        self,
        num_embeddings: int,
        num_continuous: int,
        num_outputs: int,
        num_layers: int = 2,
    ) -> List[int]:
        """
        Generates sizes of layers of neural network using heuristic.

        Args:
            num_embeddings (int): sum of counts of all categories from dataset
            num_continuous (int): number of continuous variables
            num_outputs (int): number of outputs
            num_layers (int, optional): number of layers. Defaults to 2.

        Returns:
            List[int]: sizes of layers of neural network
        """
        num_input_nodes = num_embeddings + num_continuous
        first_layer = 2 ** (num_layers - 1) * round(
            (((2 / 3) * num_input_nodes) + num_outputs) / 2 ** (num_layers - 1)
        )

        return [first_layer] + [
            int(first_layer / 2 ** n) for n in range(1, num_layers)
        ]

    def _fit_nn_model(self, nn_model, embedder_n_epochs: int) -> None:
        """
        Fits passed Learner object

        Args:
            nn_model (Learner): object to fit
            embedder_n_epochs (int): number of epochs used to train model
        """
        valley = nn_model.lr_find()
        nn_model.fit_one_cycle(n_epoch=embedder_n_epochs, lr_max=valley)

    def _embed_features(
        self, learner: Learner, tabular_pandas: TabularPandas
    ) -> pd.DataFrame:
        """
        Method used for category embedding. Extracts proper row from
        matrix which corresponds to first layer of network and treats
        them as numerical representations of categories

        Args:
            learner (Learner): Object to extract numerical representations
                of categories
            tabular_pandas (TabularPandas): Converted frame containing
                categories to embed

        Returns:
            pd.DataFrame: frame with dropped categorical variables and
                category embeddings of them joined to end.
        """
        xs: pd.DataFrame = tabular_pandas.xs
        xs_cont: pd.DataFrame = xs[learner.dls.cont_names]
        xs_cat: pd.DataFrame = xs[learner.dls.cat_names]
        for i, col in enumerate(xs_cat.columns):
            embeddings: nn.Embedding = learner.model.embeds[i]
            embedding_data: torch.Tensor = embeddings(
                tensor(xs_cat[col], dtype=torch.int64)
            )
            embedding_names: List[str] = [
                f"{col}_{j}" for j in range(embedding_data.shape[1])
            ]

            df_local: pd.DataFrame = pd.DataFrame(
                data=embedding_data,
                index=xs_cat.index,
                columns=embedding_names,
            )
            xs_cat: pd.DataFrame = xs_cat.drop(col, axis=1)
            xs_cat: pd.DataFrame = xs_cat.join(df_local)
        xs: pd.DataFrame = pd.concat([xs_cont, xs_cat], axis=1)

        return xs

