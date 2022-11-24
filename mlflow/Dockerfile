FROM python:3.9-slim
ARG internal_ip="0.0.0.0:5000"

RUN pip install --no-cache-dir mlflow
RUN pip install --no-cache-dir google-cloud-storage

EXPOSE 5000

ENV INTERNAL_IP=$internal_ip
RUN echo $INTERNAL_IP

COPY service_account_key.json .
ENV GOOGLE_APPLICATION_CREDENTIALS=service_account_key.json

CMD ls && mlflow server \
        --backend-store-uri sqlite:///mlflow/mlflow_backend_store.db \
        --default-artifact-root /mlflow/artifacts \
        --host $INTERNAL_IP