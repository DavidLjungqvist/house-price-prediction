# Schema Inference Prompt for ML Regression Pipeline

You are working inside a machine learning regression project that uses a `DataValidator` class to enforce dataset correctness before preprocessing and training.

Your task is to **generate or update `schema.yaml` automatically** based on:

* the existing validation logic in `DataValidator`
* the training dataset (`train`)
* the test dataset (`test`)
* and any existing `schema.yaml` if present

---

# Context: Validation System

The project uses the following validator:

```python
DataValidator(schema, is_train=True)
```

Validation rules include:

* required columns
* dtypes
* numeric ranges
* allowed categorical values
* uniqueness constraints

The schema is used for both training and inference, with a key difference:

* `target_column` is required during training
* `target_column` must be ignored during inference (`is_train=False`)

---

# Your Goal

Generate a complete, production-safe `schema.yaml` that satisfies:

* correctness on train/test data
* consistency with preprocessing expectations
* no overfitting to dataset noise
* no unsafe or overly strict constraints

---

# Input Data

You may receive:

* `train.csv`
* `test.csv`
* optional `original.csv`
* existing `schema.yaml`
* `preprocessing.py`

You must inspect all available datasets and code.

---

# Schema Format to Produce

Output must match this structure:

```yaml
required_columns:
  - ...

target_column: ...

dtypes:
  column_name: dtype

ranges:
  column_name:
    min: value
    max: value

allowed_values:
  column_name:
    - value1
    - value2

unique_keys:
  - column_name
```

---

# Inference Rules

## 1. required_columns

Include:

* all columns present in training data
* all columns referenced in preprocessing
* exclude only temporary engineered features if explicitly dropped later

DO NOT include:

* internal helper columns
* purely intermediate transformation columns

---

## 2. target_column

Infer from:

* preprocessing code
* or fallback: last column if clearly a regression dataset

In this dataset style, typically:

```text
SalePrice
```

---

## 3. dtypes

Infer using dataset + code consistency.

Rules:

* numeric columns → `int64` or `float64`
* categorical columns → `object`
* year fields → `int64` (do NOT convert to datetime unless explicitly done in code)
* binary flags → `object` unless encoded numerically in preprocessing

Special handling:

* NA-like strings ("NA") should NOT change dtype inference
* treat them as categorical missing markers, not real null types

---

## 4. ranges (VERY CONSERVATIVE)

Only define ranges when logically meaningful.

Allowed rules:

### Always safe to infer:

* non-negative constraints:

  * area, square footage, prices, counts

Example:

```yaml
LotArea:
  min: 0

GrLivArea:
  min: 0

SalePrice:
  min: 0
```

### Infer cautiously:

* quality scores (OverallQual, OverallCond):

  * typically bounded 1–10

### DO NOT:

* use dataset min/max as hard constraints
* invent tight upper bounds unless domain-meaningful

---

## 5. allowed_values

Only include for low-cardinality categorical variables.

Good candidates:

* Street → {Pave, Grvl}
* CentralAir → {Y, N}
* PavedDrive → {Y, P, N}
* Utilities → if constant, include single value
* SaleCondition → small fixed set

Bad candidates:

* Neighborhood (too many categories)
* Id
* any high-cardinality field

---

## 6. unique_keys

Infer only if:

* column is clearly an identifier
* AND duplicates do not exist in train/test

Typical:

```text
Id
```

If duplicates exist, do NOT enforce uniqueness.

---

## 7. Missing values handling awareness

Columns like:

* Alley
* PoolQC
* Fence
* MiscFeature
* FireplaceQu

may contain `"NA"` meaning "missing category"

Treat these as valid categorical values, NOT null schema errors.

---

## 8. Train/Test consistency rule

A constraint is valid only if:

* it holds in BOTH train and test (if available)

If a rule breaks in test:

* relax or remove it

---

# Output Requirements

You must produce:

## 1. schema.yaml (complete file)

No partial output.

Must be valid YAML.

---

## 2. schema reasoning summary

Explain:

* what was inferred
* what was preserved from existing schema
* what was removed and why

---

## 3. risk notes

List:

* overly strict constraints avoided
* uncertain inferences
* columns with ambiguous meaning

---

# Safety Principle

Prefer:

> "slightly too permissive schema"

over:

> "over-constrained schema that breaks training"

A schema that is too strict is worse than one that is slightly loose.

---

# Final Objective

The resulting schema must be:

* stable across future dataset versions
* safe for production validation
* compatible with DataValidator class
* minimal but expressive enough to catch real data errors
