# %%
from sklearn.datasets import load_iris

df, target = load_iris(as_frame=True, return_X_y=True)

# %%
df

# %%
target

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

logistic_regression = make_pipeline(StandardScaler(), LogisticRegression())
logistic_regression

# %%
import pandas as pd
from sklearn.model_selection import cross_validate, KFold

cv = KFold(n_splits=3)
cv_results = cross_validate(
    logistic_regression, df, target, cv=cv, return_train_score=True
)
cv_results = pd.DataFrame(cv_results)
cv_results[["train_score", "test_score"]]

# %%
ax = target.plot()
_ = ax.set(
    xlabel="Sample index",
    ylabel="Target value",
    title="Iris dataset target values",
)

# %%
for cv_fold_idx, (train_indices, test_indices) in enumerate(cv.split(df, target)):
    print(f"Fold {cv_fold_idx}:\n")
    print(
        f"Class counts on the train set:\n"
        f"{target.iloc[train_indices].value_counts()}"
    )
    print(
        f"Class counts on the test set:\n"
        f"{target.iloc[test_indices].value_counts()}"
    )
    print()

# %%
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=3)
cv_results = cross_validate(
    logistic_regression, df, target, cv=cv, return_train_score=True
)
cv_results = pd.DataFrame(cv_results)
cv_results[["train_score", "test_score"]]

# %%
for cv_fold_idx, (train_indices, test_indices) in enumerate(cv.split(df, target)):
    print(f"Fold {cv_fold_idx}:\n")
    print(
        f"Class counts on the train set:\n"
        f"{target.iloc[train_indices].value_counts()}"
    )
    print(
        f"Class counts on the test set:\n"
        f"{target.iloc[test_indices].value_counts()}"
    )
    print()

# %%
from sklearn.datasets import load_breast_cancer

df, target = load_breast_cancer(as_frame=True, return_X_y=True)

# %%
target.value_counts(normalize=True)

# %%
cv = KFold(n_splits=3, shuffle=True, random_state=0)
for cv_fold_idx, (train_indices, test_indices) in enumerate(cv.split(df, target)):
    print(f"Fold {cv_fold_idx}:\n")
    print(
        "Class counts on the train set:\n"
        f"{target.iloc[train_indices].value_counts(normalize=True)}\n"
    )
    print(
        f"Class counts on the test set:\n"
        f"{target.iloc[test_indices].value_counts(True)}"
    )
    print()

# %%
cv = StratifiedKFold(n_splits=3)
for cv_fold_idx, (train_indices, test_indices) in enumerate(cv.split(df, target)):
    print(f"Fold {cv_fold_idx}:\n")
    print(
        "Class counts on the train set:\n"
        f"{target.iloc[train_indices].value_counts(normalize=True)}\n"
    )
    print(
        f"Class counts on the test set:\n"
        f"{target.iloc[test_indices].value_counts(True)}"
    )
    print()

# %%
help(cross_validate)

# %%
from sklearn.datasets import load_digits

df, target = load_digits(return_X_y=True)

# %%
from sklearn.preprocessing import MinMaxScaler

logistic_regression = make_pipeline(MinMaxScaler(), LogisticRegression())

# %%
cv = KFold(n_splits=13)
cv_results = cross_validate(logistic_regression, df, target, cv=cv)
print(
    f"Mean test score: {cv_results['test_score'].mean():.3f} +/- "
    f"{cv_results['test_score'].std():.3f}"
)

# %%
cv = KFold(n_splits=13, shuffle=True, random_state=0)
cv_results = cross_validate(logistic_regression, df, target, cv=cv)
print(
    f"Mean test score: {cv_results['test_score'].mean():.3f} +/- "
    f"{cv_results['test_score'].std():.3f}"
)
# %%
for seed in range(10):
    cv = KFold(n_splits=13, shuffle=True, random_state=seed)
    cv_results = cross_validate(logistic_regression, df, target, cv=cv)
    print(
        f"Mean test score: {cv_results['test_score'].mean():.3f} +/- "
        f"{cv_results['test_score'].std():.3f}"
    )

# %%
from itertools import count
import numpy as np

# defines the lower and upper bounds of sample indices
# for each writer
writer_boundaries = [
    0,
    130,
    256,
    386,
    516,
    646,
    776,
    915,
    1029,
    1157,
    1287,
    1415,
    1545,
    1667,
    1797,
]
groups = np.zeros_like(target)
lower_bounds = writer_boundaries[:-1]
upper_bounds = writer_boundaries[1:]

for group_id, lb, up in zip(count(), lower_bounds, upper_bounds):
    groups[lb:up] = group_id

# %%
import matplotlib.pyplot as plt

plt.plot(groups)
plt.yticks(np.unique(groups))
plt.xticks(writer_boundaries, rotation=90)
plt.xlabel("Target index")
plt.ylabel("Writer index")
_ = plt.title("Underlying writer groups existing in the target")

# %%
from sklearn.model_selection import GroupKFold

cv = GroupKFold(n_splits=13)
cv_results = cross_validate(logistic_regression, df, target, groups=groups, cv=cv)
print(
    f"Mean test score: {cv_results['test_score'].mean():.3f} +/- "
    f"{cv_results['test_score'].std():.3f}"
)

# %%
