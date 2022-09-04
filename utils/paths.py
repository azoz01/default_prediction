import os

ROOT_PATH = os.path.abspath(os.getcwd())
RAW_DATA_PATH = os.path.join(ROOT_PATH, "data", "raw")
SPLITTED_DATA_PATH = os.path.join(ROOT_PATH, "data", "splitted")
NO_MISSING_DATA_PATH = os.path.join(ROOT_PATH, "data", "no_missing")
NUMERICAL_TRANSFORMED_DATA_PATH = os.path.join(
    ROOT_PATH, "data", "numerical_transformed"
)
CATEGORICAL_TRANSFORMED_DATA_PATH = os.path.join(
    ROOT_PATH, "data", "categorical_transformed"
)
DATA_TO_TRAIN_PATH = CATEGORICAL_TRANSFORMED_DATA_PATH
PIPELINES_SERIALIZED_PATH = os.path.join(ROOT_PATH, "pipelines", "serialized")
MODELS_SERIALIZED_PATH = os.path.join(ROOT_PATH, "models", "serialized")
MODELS_PARAMETERS_PATH = os.path.join(ROOT_PATH, "models", "parameters")


def main():
    print(ROOT_PATH)


if __name__ == "__main__":
    main()
