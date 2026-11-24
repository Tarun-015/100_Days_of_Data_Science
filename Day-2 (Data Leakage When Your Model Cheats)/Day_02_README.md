                            # Day 02 — Data Leakage: When Your Model Cheats

## The Problem

A machine-learning model can achieve an excellent validation or test score and still fail in production. One reason is **data leakage**: information that would not genuinely be available at prediction time influences model training or evaluation.

**Core question:** Can a model achieve an impressive test score and still fail in the real world?

## What We Learn Today

- Target leakage and train-test contamination
- Why the timing and availability of a feature matter
- How preprocessing can leak information
- Why splitting must happen before fitting transformations
- How scikit-learn Pipelines help prevent leakage
- Why a high test score is not automatically trustworthy

## Dataset

We continue with the Titanic dataset used on Day 01, loaded through Seaborn:

```python
import seaborn as sns
df = sns.load_dataset("titanic")
```

The target is `survived`. The dataset also contains `alive`, a column that directly represents the survival outcome. This makes it useful for demonstrating target leakage.

Dataset source: [Seaborn Titanic dataset](https://github.com/mwaskom/seaborn-data/blob/master/titanic.csv).

## 1. What Is Data Leakage?

Data leakage occurs when information that would not be available at prediction time is used to build or evaluate a model. It can create overly optimistic performance estimates and poor performance on genuinely new data.

Imagine predicting whether a customer will default on a loan. A feature containing the customer's eventual recovery status would reveal information from after the prediction point. It may be highly predictive, but it is not a valid input for an early-warning model.

The key question is not only **“Is this feature correlated with the target?”** It is also **“Would we know this feature when the prediction is made?”**

## 2. Major Types of Leakage

### A. Target leakage

A feature contains the target itself, a direct proxy for it, or information recorded after the outcome.

Titanic example:
- Target: `survived`
- Suspicious feature: `alive`

Using `alive` to predict `survived` is invalid because it gives the model the answer.

### B. Train-test contamination

Information from the test set influences the training process. This can happen when preprocessing or feature selection is fitted on the entire dataset before splitting.

Example of an unsafe workflow:

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Learns from all rows
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)
```

The scaler has already used information from the future test set.

### C. Time-based leakage

A model uses information from the future to predict the past or present.

Example: predicting tomorrow's sales using a feature calculated from tomorrow's actual sales.

### D. Group leakage

Records from the same person, patient, customer, device or household appear in both training and test sets when the evaluation is meant to measure performance on unseen groups.

A random row split may make the test set easier than the real deployment problem. Use group-aware splitting when the deployment goal involves new groups.

## 3. Statistical Study of the Data

The notebook investigates:

| Analysis | Why it matters |
|---|---|
| Dataset shape and data types | Understand what is being analysed |
| Missing-value counts and percentages | Identify incomplete features |
| Mean, median, standard deviation and quartiles | Understand numerical distributions |
| Target counts and proportions | Understand class balance |
| Survival rate by sex and passenger class | Explore relationships with the outcome |
| `alive` versus `survived` cross-tabulation | Detect a feature that reveals the target |
| Full-data versus training-only scaler means | See how test data can influence preprocessing |

A strong relationship is not automatically evidence of leakage. The feature's **availability at prediction time** determines whether it is legitimate.

## 4. Experiment

We compare two logistic-regression models using the same train/test split:

1. A model using legitimate passenger features.
2. A model that also uses `alive`.

The second model is deliberately invalid. Its score illustrates how target leakage can make evaluation misleading.

We then compare the statistical parameters learned by:
- A scaler fitted on the full dataset.
- A scaler fitted only on the training set.

The notebook also demonstrates a safer pipeline-based workflow.

## 5. Correct Workflow

The safe order is:

1. Define the prediction target and prediction time.
2. Split the data into training and test sets.
3. Fit imputers, scalers, encoders and feature selectors on training data only.
4. Apply the fitted transformations to validation and test data.
5. Keep the test set untouched during model selection.
6. Use time-based or group-based splitting when the problem requires it.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression())
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

The pipeline learns the scaler parameters during `fit` on the training data and applies those learned parameters to later data.

## 6. Common Mistakes

- Scaling or imputing the complete dataset before splitting.
- Selecting features using the complete dataset before cross-validation.
- Including columns created after the target event.
- Using future information in time-series features.
- Allowing the same entity to appear in both train and test when evaluating unseen entities.
- Repeatedly checking test performance and changing the model based on it.
- Assuming a high score proves the model is production-ready.

## 7. Interview Questions

1. What is data leakage, and why is it dangerous?
2. What is the difference between target leakage and train-test contamination?
3. Why should `fit_transform()` be applied to training data only?
4. What is the difference between `fit()` and `transform()`?
5. How does a scikit-learn Pipeline help prevent leakage?
6. Why can a random train-test split be inappropriate for time-series data?
7. When would you use `GroupKFold` or `GroupShuffleSplit`?
8. Can a feature have a very high correlation with the target and still be invalid?

## 8. Key Takeaways

- Leakage makes model evaluation overly optimistic.
- A feature must be available at the time the prediction is made.
- Split before fitting preprocessing transformations.
- Learn preprocessing parameters from training data only.
- Use pipelines to make safe workflows easier.
- Match the split strategy to the real-world prediction setting.
- A high score is not proof of a good model.

## References

- [Scikit-learn — Common pitfalls and data leakage](https://scikit-learn.org/stable/common_pitfalls.html)
- [Scikit-learn — Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Seaborn Titanic dataset](https://github.com/mwaskom/seaborn-data/blob/master/titanic.csv)

---

**Day 02 principle:** A model should learn from information that would genuinely be available when making its prediction — nothing more.
