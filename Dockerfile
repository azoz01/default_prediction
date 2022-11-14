FROM python:3.9

ENV GITHUB_REPO_URL="https://github.com/azoz01/default_prediction"
ENV CURRENT_MODEL="xgboost_final_2.pkl"

WORKDIR /code
RUN git clone ${GITHUB_REPO_URL}
WORKDIR /code/default_prediction
RUN git checkout develop

RUN pip install --no-cache-dir -r requirements.txt
RUN dvc pull -r google_storage resources/preprocessing/serialized/*
RUN dvc pull -r google_storage resources/models/serialized/*

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "80"]