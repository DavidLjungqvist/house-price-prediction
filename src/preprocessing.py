import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


TARGET = "SalePrice"

# Columns where NA means "None" / absence of feature (not actually missing)
NA_MEANS_NONE = [
    "Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
    "BsmtFinType2", "FireplaceQu", "GarageType", "GarageFinish",
    "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature"
]

# Ordinal features with a meaningful order (low → high)
ORDINAL_MAPPINGS = {
    "ExterQual": ["Po", "Fa", "TA", "Gd", "Ex"],
    "ExterCond": ["Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtQual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtCond": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtExposure": ["None", "No", "Mn", "Av", "Gd"],
    "BsmtFinType1": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "BsmtFinType2": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "HeatingQC": ["Po", "Fa", "TA", "Gd", "Ex"],
    "KitchenQual": ["Po", "Fa", "TA", "Gd", "Ex"],
    "FireplaceQu": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "GarageFinish": ["None", "Unf", "RFn", "Fin"],
    "GarageQual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "GarageCond": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "PoolQC": ["None", "Fa", "TA", "Gd", "Ex"],
    "Fence": ["None", "MnWw", "GdWo", "MnPrv", "GdPrv"],
    "Functional": ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
    "LandSlope": ["Sev", "Mod", "Gtl"],
    "LotShape": ["IR3", "IR2", "IR1", "Reg"],
    "PavedDrive": ["N", "P", "Y"],
    "Utilities": ["ELO", "NoSeWa", "NoSewr", "AllPub"],
    "CentralAir": ["N", "Y"],
}

# Columns to drop (not useful for modeling)
DROP_COLS = ["Id"]


def identify_column_types(df):
    """Split columns into numeric and categorical (excluding target and drop cols)."""
    cols = [c for c in df.columns if c not in DROP_COLS + [TARGET]]
    numeric_cols = [c for c in cols if df[c].dtype in ["int64", "float64"]]
    categorical_cols = [c for c in cols if df[c].dtype == "object"]
    return numeric_cols, categorical_cols


def fill_na_as_none(df):
    """For columns where NA means 'no feature', fill with 'None'."""
    df = df.copy()
    for col in NA_MEANS_NONE:
        if col in df.columns:
            df[col] = df[col].fillna("None")
    return df


def build_preprocessor(df):
    """Build a sklearn ColumnTransformer for the dataset.

    Args:
        df: A DataFrame (training set) used to determine column types.

    Returns:
        A fitted-ready ColumnTransformer pipeline.
    """
    numeric_cols, categorical_cols = identify_column_types(df)

    ordinal_cols = [c for c in categorical_cols if c in ORDINAL_MAPPINGS]
    nominal_cols = [c for c in categorical_cols if c not in ORDINAL_MAPPINGS]

    # Numeric: impute with median
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    # Ordinal: encode with order
    ordinal_categories = [ORDINAL_MAPPINGS[col] for col in ordinal_cols]
    ordinal_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(categories=ordinal_categories, handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    # Nominal: one-hot encode
    nominal_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("ord", ordinal_transformer, ordinal_cols),
            ("nom", nominal_transformer, nominal_cols),
        ],
        remainder="drop",
    )

    return preprocessor


def preprocess(train_df, test_df=None):
    """Run the full preprocessing pipeline.

    Args:
        train_df: Raw training DataFrame.
        test_df: Optional raw test DataFrame.

    Returns:
        If test_df is None: (X_train, y_train, preprocessor)
        If test_df provided: (X_train, y_train, X_test, preprocessor)
    """
    train_df = fill_na_as_none(train_df)
    y_train = np.log1p(train_df[TARGET]) if TARGET in train_df.columns else None

    preprocessor = build_preprocessor(train_df)
    X_train = preprocessor.fit_transform(train_df)

    if test_df is not None:
        test_df = fill_na_as_none(test_df)
        X_test = preprocessor.transform(test_df)
        return X_train, y_train, X_test, preprocessor

# Add this new feature engineering function after preprocess()

def create_product_feature(df, col1_name, col2_name):
    """Create a product feature from two numeric columns.

    Args:
        df: A pandas DataFrame.
        col1_name: Name of the first column to multiply.
        col2_name: Name of the second column to multiply.

    Returns:
        A new column name and the resulting product column (if applicable).
    """
    if col1_name not in df.columns or col2_name not in df.columns:
        return None

    # Check if both columns are numeric
    if df[col1_name].dtype in ["int64", "float64"] and \
            df[col2_name].dtype in ["int64", "float64"]:
        new_col_name = f"{col1_name}*{col2_name}"
        df[new_col_name] = df[col1_name] * df[col2_name]
        return new_col_name

    return None


def create_interaction_features(df, col_pairs=None):
    """Create multiple product/interaction features from given column pairs.

    Args:
        df: A pandas DataFrame.
        col_pairs: List of tuples with (col1_name, col2_name), or None for auto-detection.

    Returns:
        DataFrame with new interaction columns added.
    """
    if col_pairs is None:
        # Auto-detect numeric column pairs to create interactions from
        cols = [c for c in df.columns if df[c].dtype in ["int64", "float64"]]
        col_pairs = [(cols[i], cols[i+1]) for i in range(len(cols)-1)]

    original_cols = set(df.columns)

    for col1, col2 in col_pairs:
        new_col_name = f"{col1}*{col2}"

        # Add product column if both are numeric and don't already exist
        if col1 in original_cols and col2 in original_cols and \
                col1 != col2:  # Avoid squaring the same column by default
            df[new_col_name] = df[col1] * df[col2]

    return df