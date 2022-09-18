from typing import Tuple
from pandas import DataFrame
from sklearn.base import TransformerMixin
import pandas as pd
from utils.logging import pipeline_logger
from imblearn.over_sampling import RandomOverSampler
from fastai.tabular.all import *


class CategoryEmbedder(TransformerMixin):
    def fit(self, X: DataFrame, y: DataFrame, embedder_n_epochs=20, **kwargs):
        X = self._categorize_categorical_variables(X)
        X, y = self._resample_data(X, y)
        tabular_pandas: TabularPandas = self._get_tabular_pandas(
            X=X, y=y, target_name="TARGET"
        )
        self.nn_model = self._prepare_nn_model(tabular_pandas=tabular_pandas)
        self._fit_nn_model(
            nn_model=self.nn_model, embedder_n_epochs=embedder_n_epochs
        )
        return self

    def transform(self, X: DataFrame, y: DataFrame = None, **kwargs):
        X = self._categorize_categorical_variables(X)
        tabular_pandas: TabularPandas = self._get_tabular_pandas(X)
        X_embedded: DataFrame = self._embed_features(
            self.nn_model, tabular_pandas
        )
        return X_embedded

    def _categorize_categorical_variables(self, X: DataFrame) -> DataFrame:
        pipeline_logger.info("Categorizing non-numerical columns inside frame")
        X = X.copy()
        X[X.select_dtypes(exclude=["float64"]).columns.values] = X[
            X.select_dtypes(exclude=["float64"]).columns.values
        ].astype("category")
        return X

    def _resample_data(
        self, X: DataFrame, y: DataFrame
    ) -> Tuple[DataFrame, DataFrame]:
        pipeline_logger.info("Resampling data")
        ros = RandomOverSampler()
        X_resampled, y_resampled = ros.fit_resample(X, y)
        return X_resampled, y_resampled

    def _get_tabular_pandas(
        self,
        X: DataFrame,
        y: DataFrame = None,
        target_name: str = None,
        valid_sample_frac: float = 0.3,
    ) -> TabularPandas:
        if y is not None:
            df = pd.concat([X, y], axis=1)
        else:
            df = X

        categorical_columns = X.select_dtypes(
            exclude=["float64"]
        ).columns.values.tolist()
        continuous_columns = X.select_dtypes(
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

    def _prepare_nn_model(self, tabular_pandas: TabularPandas):
        num_embeddings = sum(n for _, n in get_emb_sz(tabular_pandas))
        num_classes = tabular_pandas.ys.nunique().values[0]
        continuous_columns = tabular_pandas.cont_names
        layers = self._get_default_nn_layers(
            num_embeddings,
            num_continuous=len(continuous_columns),
            num_outputs=num_classes,
            num_layers=3,
        )
        config = tabular_config(ps=[0.001, 0.01, 0.01], embed_p=0.04)

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
    ) -> list[int]:
        num_input_nodes = num_embeddings + num_continuous
        first_layer = 2 ** (num_layers - 1) * round(
            (((2 / 3) * num_input_nodes) + num_outputs) / 2 ** (num_layers - 1)
        )

        return [first_layer] + [
            int(first_layer / 2 ** n) for n in range(1, num_layers)
        ]

    def _fit_nn_model(self, nn_model, embedder_n_epochs: int) -> None:
        valley = nn_model.lr_find()
        nn_model.fit_one_cycle(n_epoch=embedder_n_epochs, lr_max=valley)

    def _embed_features(
        self, learner, tabular_pandas: TabularPandas
    ) -> pd.DataFrame:
        xs = tabular_pandas.xs
        xs_cont = xs[learner.dls.cont_names]
        xs_cat = xs[learner.dls.cat_names]
        for i, col in enumerate(xs_cat.columns):
            embeddings = learner.model.embeds[i]
            embedding_data = embeddings(tensor(xs_cat[col], dtype=torch.int64))
            embedding_names = [
                f"{col}_{j}" for j in range(embedding_data.shape[1])
            ]

            df_local = pd.DataFrame(
                data=embedding_data,
                index=xs_cat.index,
                columns=embedding_names,
            )
            xs_cat = xs_cat.drop(col, axis=1)
            xs_cat = xs_cat.join(df_local)
        xs = pd.concat([xs_cont, xs_cat], axis=1)

        return xs

