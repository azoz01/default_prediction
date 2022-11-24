import logging
import mlflow
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from models.lib.utils.metrics import get_metrics
from utils.parameters import get_data_path
from utils.io import load_data

logger = logging.getLogger(__name__)

DATA_PATH_NAME = "reduced"


def main() -> None:
    with mlflow.start_run(run_name="neural_network"):
        data_path = get_data_path(DATA_PATH_NAME)
        logger.info("Loading data")
        X_train, y_train, X_valid, y_valid, X_test, y_test = load_data(
            path=data_path
        )
        params = {
            "hidden_layers": 4,
            "hidden_layer_size": 200,
            "learning_rate": 1e-3,
            "epochs": 5,
            "batch_size": 1024,
            "threshold": y_train.mean().values[0],
        }
        inp = tf.keras.layers.Input(shape=(X_train.shape[1],))
        x = BatchNormalization()(inp)
        for _ in range(params["hidden_layers"]):
            x = Dense(params["hidden_layer_size"], activation="relu")(x)
            x = BatchNormalization()(x)
        x = Dense(1, activation="sigmoid")(x)
        model = Model(inputs=inp, outputs=x)
        model.compile(
            optimizer=Adam(params["learning_rate"]),
            loss="binary_crossentropy",
            metrics=[AUC(name="AUC")],
        )
        model.summary()

        for name, value in params.items():
            mlflow.log_param(name, value)

        logger.info("Fitting model to train data")
        model.fit(
            X_train,
            y_train,
            validation_data=(X_valid, y_valid),
            epochs=params["epochs"],
            batch_size=params["batch_size"],
        )

        def predict(X: np.ndarray) -> np.ndarray:
            return model.predict(X) >= params["threshold"]

        logger.info("Evaluating model")
        train_metrics = get_metrics(
            model, X_train, y_train, predict_fun=predict
        )
        logger.info(f"Metrics on train set: {train_metrics}")
        mlflow.log_param("type", "neural_network")
        for metric, value in train_metrics.items():
            mlflow.log_metric(f"train_{metric}", value)
        logger.info(
            "Metrics on validation set: "
            f"{get_metrics(model, X_valid, y_valid, predict_fun=predict)}"
        )
        test_metrics = get_metrics(model, X_test, y_test, predict_fun=predict)
        logger.info(f"Metrics on test set: {test_metrics}")
        for metric, value in test_metrics.items():
            mlflow.log_metric(f"test_{metric}", value)

        logger.info("Saving model")
        model.save("resources/models/serialized/neural_network")
        logger.info("Publishing artifacts")


if __name__ == "__main__":
    main()
