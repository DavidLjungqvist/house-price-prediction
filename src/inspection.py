import matplotlib.pyplot as plt
import seaborn as sns



### Placeholder ###
def report_missing_values(df_train):
    missing_values = df_train.isnull().sum()
    missing_values = missing_values[missing_values > 0]

    if not missing_values.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=missing_values.index, y=missing_values.values, palette="viridis")
        plt.xticks(rotation=90)
        plt.xlabel("Features")
        plt.ylabel("Missing Values")
        plt.title("Missing Values per Feature")
        plt.tight_layout()
        plt.show()
    else:
        print("✅ No missing values found in the dataset.")

def print_categories(df, cols):
    # print all existing categories and their percentages.
    for col in cols:
        print(f"\nColumn: `{col}`")
        number_of = df[col].value_counts(dropna=False, normalize=True) * 100
        for name, percentage in number_of.items():
            print(f"  {name:13}: {percentage:6.2f}%")
### Placeholder ###