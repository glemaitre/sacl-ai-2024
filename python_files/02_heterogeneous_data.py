# %%
import pandas as pd

df = pd.read_csv("../datasets/penguins.csv")
df

# %%
feature_names = [
    "Region",
    "Island",
    "Culmen Length (mm)",
    "Culmen Depth (mm)",
    "Flipper Length (mm)",
    "Body Mass (g)",
    "Sex",
]
target_name = "Species"
X = df[feature_names]
y = df[target_name]

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# %%
from sklearn.linear_model import LogisticRegression

logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)

# %%
X

# %%
X.info()

# %%
X_numeric = X.select_dtypes(include="number")

# %%
logistic_regression.fit(X_numeric, y)

# %%
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

logistic_regression = make_pipeline(SimpleImputer(), LogisticRegression())
logistic_regression.fit(X_numeric, y)

# %%
logistic_regression[-1].n_iter_

# %%
logistic_regression.get_params()
# %%
logistic_regression.set_params(logisticregression__max_iter=10_000)
logistic_regression.fit(X_numeric, y)

# %%
logistic_regression[-1].n_iter_

# %%
from sklearn.preprocessing import StandardScaler

logistic_regression = make_pipeline(
    StandardScaler(), SimpleImputer(), LogisticRegression()
)
logistic_regression.fit(X_numeric, y)
logistic_regression[-1].n_iter_

# %%
logistic_regression_without_scaling = make_pipeline(
    SimpleImputer(), LogisticRegression(max_iter=10_000)
).fit(X_numeric, y)

# %%
X_categorical = X.select_dtypes(exclude="number")
X_categorical

# %%
from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder().set_output(transform="pandas")
X_encoded = ordinal_encoder.fit_transform(X_categorical)
X_encoded

# %%
from sklearn.preprocessing import OneHotEncoder

one_hot_encoder = OneHotEncoder(sparse_output=False).set_output(transform="pandas")
X_encoded = one_hot_encoder.fit_transform(X_categorical)
X_encoded

# %%
logistic_regression = make_pipeline(
    OneHotEncoder(), LogisticRegression()
)
logistic_regression.fit(X_categorical, y)

# %%
from sklearn.compose import make_column_transformer, make_column_selector as selector

preprocessor = make_column_transformer(
    (make_pipeline(StandardScaler(), SimpleImputer()), selector(dtype_include="number")),
    (OneHotEncoder(), selector(dtype_exclude="number")),
)
logistic_regression = make_pipeline(preprocessor, LogisticRegression())
logistic_regression

# %%
logistic_regression.fit(X_train, y_train)

# %%
y.value_counts()

# %%
from sklearn.metrics import classification_report

print(
    classification_report(y_test, logistic_regression.predict(X_test))
)

# %%
from sklearn.model_selection import cross_validate

cv_results = cross_validate(logistic_regression, X, y, return_train_score=True)
cv_results = pd.DataFrame(cv_results)
cv_results

# %%
logistic_regression.get_params()

# %%
logistic_regression.set_params(
    columntransformer__onehotencoder__handle_unknown="ignore"
)

# %%
cv_results = cross_validate(logistic_regression, X, y, return_train_score=True)
cv_results = pd.DataFrame(cv_results)
cv_results

# %%
preprocessor = make_column_transformer(
    # (make_pipeline(StandardScaler(), SimpleImputer()), selector(dtype_include="number")),
    (OneHotEncoder(handle_unknown="ignore"), selector(dtype_exclude="number")),
)
logistic_regression = make_pipeline(preprocessor, LogisticRegression())

# %%
cv_results = cross_validate(logistic_regression, X, y, return_train_score=True)
cv_results = pd.DataFrame(cv_results)
cv_results

# %%
# %%
preprocessor = make_column_transformer(
    (make_pipeline(StandardScaler(), SimpleImputer()), selector(dtype_include="number")),
    # (OneHotEncoder(handle_unknown="ignore"), selector(dtype_exclude="number")),
)
logistic_regression = make_pipeline(preprocessor, LogisticRegression())

# %%
cv_results = cross_validate(logistic_regression, X, y, return_train_score=True)
cv_results = pd.DataFrame(cv_results)
cv_results

# %%
