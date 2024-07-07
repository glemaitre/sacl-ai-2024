# %%
import pandas as pd

df = pd.read_csv("../datasets/penguins.csv")
df

# %%
feature_names = [
    "Region",
    "Island",
    "Culmen Depth (mm)",
    "Flipper Length (mm)",
    "Body Mass (g)",
    "Sex",
]
target_name = "Species"
X = df[feature_names]
y = df[target_name]

categorical_columns = X.select_dtypes(include="object").columns
X[categorical_columns] = X[categorical_columns].astype("category")

# %%
from skrub import tabular_learner
from sklearn.linear_model import LogisticRegression

logistic_regression = tabular_learner(LogisticRegression())
logistic_regression

# %%
import pandas as pd
from sklearn.model_selection import cross_validate

cv_results = cross_validate(logistic_regression, X, y, cv=5, return_train_score=True)
cv_results = pd.DataFrame(cv_results)
cv_results[["train_score", "test_score"]]

# %%
logistic_regression.get_params()

# %%
from scipy.stats import loguniform
from sklearn.model_selection import RandomizedSearchCV

param_distributions = {
    "simpleimputer__strategy": ["mean", "median", "most_frequent"],
    "logisticregression__C": loguniform(1e-3, 1e3),
}
tuned_model = RandomizedSearchCV(
    logistic_regression,
    param_distributions=param_distributions,
    n_iter=10,
    cv=5,
    random_state=0,
)
tuned_model

# %%
cv_results = cross_validate(tuned_model, X, y, cv=5, return_estimator=True)
cv_results = pd.DataFrame(cv_results)
cv_results

# %%
for estimator in cv_results["estimator"]:
    print(estimator.best_params_)

# %%
cv_results = cross_validate(
    tuned_model, X, y, cv=5, scoring="neg_log_loss", return_estimator=True
)
cv_results = pd.DataFrame(cv_results)
cv_results

# %%
for estimator in cv_results["estimator"]:
    print(estimator.best_params_)

# %%
