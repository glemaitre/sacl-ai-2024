# %%
import pandas as pd

df = pd.read_csv("../datasets/penguins_regression.csv")
df

# %%
ax = df.plot.scatter(x="Flipper Length (mm)", y="Body Mass (g)")
_ = ax.set_title("Penguin Flipper Length vs Body Mass")

# %%
X = df[["Flipper Length (mm)"]]
y = df["Body Mass (g)"]

# %%
print(f"The dimensions of X are: {X.shape}")
print(f"The dimensions of y are: {y.shape}")

# %%
from sklearn.linear_model import LinearRegression

linear_regression = LinearRegression()
linear_regression.fit(X, y)

# %%
linear_regression.coef_, linear_regression.intercept_

# %%
import numpy as np

X_to_infer = pd.DataFrame({"Flipper Length (mm)": np.linspace(175, 230, 100)})
y_pred = linear_regression.predict(X_to_infer)

# %%
ax = df.plot.scatter(x="Flipper Length (mm)", y="Body Mass (g)")
ax.set_title("Penguin Flipper Length vs Body Mass")
ax.plot(X_to_infer, y_pred, linewidth=3, color="tab:orange", label="Linear Regression")
_ = ax.legend()

# %%
from sklearn.metrics import mean_squared_error

error = mean_squared_error(y, linear_regression.predict(X))
print(f"The mean squared error is: {error:.2f}")

# %%
from sklearn.linear_model import QuantileRegressor

quantile_regression = QuantileRegressor(alpha=0)
quantile_regression.fit(X, y)
y_pred_quantile = quantile_regression.predict(X_to_infer)
quantile_regression.coef_, quantile_regression.intercept_

# %%
ax = df.plot.scatter(x="Flipper Length (mm)", y="Body Mass (g)")
ax.set_title("Penguin Flipper Length vs Body Mass")
ax.plot(X_to_infer, y_pred, linewidth=3, color="tab:orange", label="Linear Regression")
ax.plot(
    X_to_infer,
    y_pred_quantile,
    linewidth=3,
    color="tab:red",
    label="Quantile Regression",
)
_ = ax.legend()

# %%
error = mean_squared_error(y, linear_regression.predict(X))
print(f"The mean squared error of LinearRegression is: {error:.2f}")
error = mean_squared_error(y, quantile_regression.predict(X))
print(f"The mean squared error of QuantileRegression is: {error:.2f}")

# %%
from sklearn.metrics import median_absolute_error

error = median_absolute_error(y, linear_regression.predict(X))
print(f"The median absolute error of LinearRegression is: {error:.2f}")
error = median_absolute_error(y, quantile_regression.predict(X))
print(f"The median absolute error of QuantileRegression is: {error:.2f}")

# %%
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

polynomial_features = PolynomialFeatures(degree=5).set_output(transform="pandas")
standard_scaler = StandardScaler().set_output(transform="pandas")
X_scaled = standard_scaler.fit_transform(X)
X_poly = polynomial_features.fit_transform(X_scaled)
X_poly

# %%
linear_regression.fit(X_poly, y)
X_to_infer_poly = polynomial_features.transform(
    standard_scaler.transform(X_to_infer)
)
y_pred_poly = linear_regression.predict(X_to_infer_poly)

# %%
ax = df.plot.scatter(x="Flipper Length (mm)", y="Body Mass (g)")
ax.set_title("Penguin Flipper Length vs Body Mass")
ax.plot(
    X_to_infer, y_pred_poly, linewidth=3, color="tab:orange", label="Linear Regression"
)
_ = ax.legend()

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# %%
from sklearn.metrics import mean_absolute_error

linear_regression.fit(X_train, y_train)
y_pred = linear_regression.predict(X_test)
print(
    "Mean absolute error on the training set: "
    f"{mean_absolute_error(y_train, linear_regression.predict(X_train)):.2f}"
)
print(
    "Mean absolute error on the testing set: "
    f"{mean_absolute_error(y_test, linear_regression.predict(X_test)):.2f}"
)

# %%
from sklearn.model_selection import cross_validate, KFold

cv = KFold(n_splits=5, shuffle=True, random_state=0)
cv_results = cross_validate(
    linear_regression,
    X,
    y,
    cv=cv,
    scoring="neg_mean_absolute_error",
    return_train_score=True,
)
cv_results = pd.DataFrame(cv_results)
cv_results

# %%
cv_results["test_score"] = -cv_results["test_score"]
cv_results["train_score"] = -cv_results["train_score"]
cv_results[["train_score", "test_score"]].describe()

# %%
cv_results = cross_validate(
    linear_regression,
    X_poly,
    y,
    cv=cv,
    scoring="neg_mean_absolute_error",
    return_train_score=True,
)
cv_results = pd.DataFrame(cv_results)
cv_results["test_score"] = -cv_results["test_score"]
cv_results["train_score"] = -cv_results["train_score"]
cv_results[["train_score", "test_score"]].describe()

# %%
from sklearn.pipeline import make_pipeline

linear_regression_poly = make_pipeline(
    StandardScaler(), PolynomialFeatures(degree=5), LinearRegression()
)
linear_regression_poly

# %%
linear_regression_poly.fit(X_train, y_train)
linear_regression_poly[-1].coef_, linear_regression_poly[-1].intercept_

# %%
print(
    "Mean absolute error on the training set: "
    f"{mean_absolute_error(y_train, linear_regression_poly.predict(X_train)):.2f}"
)
print(
    "Mean absolute error on the testing set: "
    f"{mean_absolute_error(y_test, linear_regression_poly.predict(X_test)):.2f}"
)

# %%
cv_results = cross_validate(
    linear_regression_poly,
    X,
    y,
    cv=cv,
    scoring="neg_mean_absolute_error",
    return_train_score=True,
)
cv_results = pd.DataFrame(cv_results)
cv_results["test_score"] = -cv_results["test_score"]
cv_results["train_score"] = -cv_results["train_score"]
cv_results[["train_score", "test_score"]].describe()

# %%
from sklearn.model_selection import RepeatedKFold

cv = RepeatedKFold(n_splits=5, n_repeats=100, random_state=0)
cv_results = cross_validate(
    linear_regression,
    X,
    y,
    cv=cv,
    scoring="neg_mean_absolute_error",
    return_train_score=True,
)
cv_results = pd.DataFrame(cv_results)
cv_results

# %%
cv_results["test_score"] = -cv_results["test_score"]
cv_results["train_score"] = -cv_results["train_score"]

# %%
ax = cv_results[["train_score", "test_score"]].plot.hist(bins=10, alpha=0.7)
_ = ax.set_xlim(0, 500)

# %%
cv_results[["train_score", "test_score"]].describe()

# %%
from sklearn.pipeline import make_pipeline

linear_regression_poly = make_pipeline(
    PolynomialFeatures(degree=10), LinearRegression()
)
cv_results_poly = cross_validate(
    linear_regression_poly,
    X,
    y,
    cv=cv,
    scoring="neg_mean_absolute_error",
    return_train_score=True,
)
cv_results_poly = pd.DataFrame(cv_results_poly)
cv_results_poly["test_score"] = -cv_results_poly["test_score"]
cv_results_poly["train_score"] = -cv_results_poly["train_score"]

# %%
test_scores = pd.concat([
    cv_results[["test_score"]].add_prefix("LinearRegression_"),
    cv_results_poly[["test_score"]].add_prefix("PolynomialRegression_"),
], axis=1)
ax = test_scores.plot.hist(bins=10, alpha=0.7)
_ = ax.set_xlim(0, 500)

# %%
linear_regression.fit(X_train, y_train)

# %%
y_pred = linear_regression.predict(X_test)
error = mean_absolute_error(y_test, y_pred)
print(f"The mean absolute error is: {error:.2f}")

# %%
n_bootstraps = 1_000
errors = []
for _ in range(n_bootstraps):
    bootstrap_indices = np.random.choice(len(y_test), size=len(y_test), replace=True)
    errors.append(mean_absolute_error(
        y_test.to_numpy()[bootstrap_indices], y_pred[bootstrap_indices])
    )
errors = np.array(errors)

# %%
print(f"The error is estimated to be {errors.mean():.2f} +/- {errors.std():.2f}")

# %%
