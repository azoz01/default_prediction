import sys
import os

sys.path.append(os.path.abspath(os.getcwd()))

import pandas as pd

import utils.paths as paths

if __name__ == "__main__":
    PATHS_WITHOUT_EXTENSIONS: list[str] = [
        os.path.join(paths.RAW_DATA_PATH, "application_data"),
        os.path.join(paths.RAW_DATA_PATH, "raw", "previous_application"),
    ]
    in_paths: list[str] = [path + ".csv" for path in PATHS_WITHOUT_EXTENSIONS]
    out_paths: list[str] = [
        path + ".parquet" for path in PATHS_WITHOUT_EXTENSIONS
    ]

    for in_path, out_path in zip(in_paths, out_paths):
        df = pd.read_csv(in_path)
        df.to_parquet(out_path)
