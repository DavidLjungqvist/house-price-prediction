# Modules
from src.data_loader import *
from src.inspection import *
from src.preprocessing import validate_data

import yaml

def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    paths = load_data(config)
    dfs = read_data(**paths)
    train_df = dfs.get("train")
    test_df = dfs.get("test")
    original_df = dfs.get("original")
    with open("schema.yaml") as f:
        schema = yaml.safe_load(f)
    validate_data(schema, train_df)
    # report_missing_values(train_df)
    # report_cardinality(train_df)
    # report_class_balance(train_df, "Alley")


main()
