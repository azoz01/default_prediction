import os

ROOT_PATH = os.path.abspath(os.getcwd())
RAW_DATA_PATH = os.path.join(ROOT_PATH, "data", "raw")
SPLITTED_DATA_PATH = os.path.join(ROOT_PATH, "data", "splitted")
NO_MISSING_DATA_PATH = os.path.join(ROOT_PATH, "data", "no_missing")
SERIALIZED_PATH = os.path.join(ROOT_PATH, "serialized")


def main():
    print(ROOT_PATH)


if __name__ == "__main__":
    main()
