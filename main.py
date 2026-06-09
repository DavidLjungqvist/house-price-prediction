# Modules
from src.data_loader import *
from src.inspection import *

import yaml

def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    paths = load_data(config)
    dfs = read_data(**paths)
    train_df = dfs.get("train")
    test_df = dfs.get("test")
    original_df = dfs.get("original")
    report_cardinality(train_df)

main()
