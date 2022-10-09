import pickle as pkl

pipelines = []

for path in pipelines_paths:
    with open(path, "rb") as f:
        pipelines.append(pkl.load(f))

print(pipelines)

from sklearn.pipeline import make_pipeline

pipelines_to_pass = list(zip(pipelines_paths, pipelines))
pipeline = make_pipeline(pipelines_to_pass)
print(pipeline)

import pandas as pd

data = pd.read_parquet("resources/data/raw/application_data.parquet")
row = data.loc[1:3]
print(row)


def transform_data(X):
    for pipeline in pipelines:
        X = pipeline.transform(X)
    return X


transformed = transform_data(row)
print(transformed)

with open(model_path, "rb") as f:
    model = pkl.load(f)

print(model.predict_proba(transformed)[:, 1])
