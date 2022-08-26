import os

ROOT_PATH = os.path.abspath(os.getcwd())
RAW_DATA_PATH = os.path.join(ROOT_PATH, "data", "raw")
SPLITTED_DATA_PATH = os.path.join(ROOT_PATH, "data", "splitted")


def main():
    print(ROOT_PATH)


if __name__ == "__main__":
    main()
