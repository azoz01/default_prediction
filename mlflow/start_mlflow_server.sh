#/bin/bash

mlflow server \
    --backend-store-uri sqlite:///mlflow/mlflow_backend_store.db \
    --default-artifact-root gs://default_prediction/mlflow \
    --host 0.0.0.0
