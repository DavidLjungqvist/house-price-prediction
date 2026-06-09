import matplotlib.pyplot as plt
import seaborn as sns



### WORK VERY MUCH IN PROGRESS ###
def report_missing_values(df_train):    # Now working
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
        print("Missing Values Count:")
        # for i in range(missing_values.size):
        #     print(f"{missing_values.index[i]}: {missing_values.values[i]}")
        for feature, count in zip(missing_values.index, missing_values.values):
            print(f"{feature}: {count}")
    else:
        print("✅ No missing values found in the dataset.")

def print_categories(df, cols):
    # print all existing categories and their percentages.
    for col in cols:
        print(f"\nColumn: `{col}`")
        number_of = df[col].value_counts(dropna=False, normalize=True) * 100
        for name, percentage in number_of.items():
            print(f"  {name:13}: {percentage:6.2f}%")

def report_cardinality(df, threshold=20):
    """Report the number of unique values per categorical column.

    Columns exceeding the threshold are flagged as high-cardinality.
    """
    cat_cols = df.select_dtypes(include="object").columns
    if cat_cols.size:
        # cardinality = df[cat_cols].value_counts(dropna=False, normalize=True).sum()
        cardinality = cat_cols.value_counts(dropna=False, normalize=False)
        cat_cols = df[cat_cols]
        print(f"{'Column':<20} {'Unique':>8}  {'Status'}")
        print("-" * 42)
        for col in cat_cols:
            n_unique = df[col].nunique()
            flag = "⚠️  HIGH" if n_unique > threshold else ""
            print(f"{col:<20} {n_unique:>8}  {flag}")
        plt.figure(figsize=(10, 6))
        sns.barplot(x=cat_cols.index, y=cat_cols.values, palette="viridis")
        plt.xticks(rotation=90)
        plt.xlabel("Features")
        plt.ylabel("Missing Values")
        plt.title("Missing Values per Feature")
        plt.tight_layout()
        plt.show()
    # Check if no categorical exists here


def report_class_balance(df, col):
    """Report the distribution of values in a column (e.g. the target).

    Shows counts and percentages, plus a bar chart.
    """
    counts = df[col].value_counts()
    total = len(df)

    print(f"Class balance for `{col}` ({total} samples):\n")
    for value, count in counts.items():
        pct = count / total * 100
        print(f"  {value}: {count} ({pct:.1f}%)")

    plt.figure(figsize=(10, 5))
    sns.countplot(x=col, data=df, order=counts.index, palette="viridis")
    plt.xticks(rotation=90)
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.title(f"Class Balance: {col}")
    plt.tight_layout()
    plt.show()

