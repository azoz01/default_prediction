FROM python:3.9

ENV CURRENT_MODEL="xgboost_final_2.pkl"

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
COPY ./models /code/models
COPY ./pipelines /code/pipelines
COPY ./utils /code/utils

RUN pip install -r requirements.txt

COPY ./api /code/api

RUN mkdir -p /code/resources/models/serialized

COPY ./resources/models/serialized/${CURRENT_MODEL} /code/resources/models/serialized

RUN mkdir -p /code/resources/pipelines/serialized

COPY ./resources/pipelines/serialized/* /code/resources/pipelines/serialized

RUN ls /code

WORKDIR /code

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "80"]