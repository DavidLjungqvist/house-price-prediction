import pandas as pd
import yaml

def load_data(config):
    paths = config["paths"]
    use_original = config["use_original_data"]
    if not use_original:
        del paths["original"] # Main issue: load_data mutates the original config Safer: paths = config["paths"].copy(). Then edits won't affect the original config object. -gpt
    return paths

def read_data(**dfs_to_load): # Function iterates through dict of "df_names" and coresponding file paths to load in CSV's as dataframes
    dfs = {}
    for df in dfs_to_load:
        try:
            loaded_df = pd.read_csv(dfs_to_load[df]) # dfs_to_load[df] becomes our filepath for this iteration i.e loaded_df = read_csv("data/raw/train.csv")
            dfs[df] = loaded_df # dictionary "dfs" gets a new key equal to the key in DATA_PATHS, and a value equal to 
        except FileNotFoundError:
            print(f"{dfs_to_load[df]} file not found")
    if dfs:
        return dfs

def add_original_data(df_train, df_original):
    pass
        

def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    paths = load_data(config)
    dfs = read_data(**paths)
    train_df = dfs.get("train")
    test_df = dfs.get("test")
    original_df = dfs.get("original")
    #print(train_df, test_df, original_df)
    #combined_df = add_original_data(train_df, original_df)

if __name__ == "__main__":
    main()