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
classes_to_keep = [
    "Adelie Penguin (Pygoscelis adeliae)",
    "Chinstrap penguin (Pygoscelis antarctica)",
]
X = X[y.isin(classes_to_keep)]
y = y[y.isin(classes_to_keep)]

# %%
from sklearn.ensemble import HistGradientBoostingClassifier

hist_gradient_boosting = HistGradientBoostingClassifier(
    categorical_features="from_dtype"
)

# %%
import pandas as pd
from sklearn.model_selection import cross_validate

cv_results = cross_validate(hist_gradient_boosting, X, y, cv=5, return_train_score=True)
cv_results = pd.DataFrame(cv_results)
cv_results[["train_score", "test_score"]]

# %%
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
hist_gradient_boosting.fit(X_train, y_train)

print(classification_report(y_test, hist_gradient_boosting.predict(X_test)))

# %%
from sklearn.metrics import PrecisionRecallDisplay

display = PrecisionRecallDisplay.from_estimator(hist_gradient_boosting, X_test, y_test)
_ = display.ax_.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curve")

# %%
y_pred = hist_gradient_boosting.predict_proba(X_test)
y_pred

# %%
y_pred.sum(axis=1)

# %%
hist_gradient_boosting.classes_[y_pred.argmax(axis=1)]

# %%
(
    hist_gradient_boosting.classes_[y_pred.argmax(axis=1)]
    == hist_gradient_boosting.predict(X_test)
)

# %%
from sklearn.model_selection import FixedThresholdClassifier
from sklearn.metrics import precision_score, recall_score

classifier = FixedThresholdClassifier(
    hist_gradient_boosting, pos_label=hist_gradient_boosting.classes_[1]
).fit(X_train, y_train)

precision = precision_score(
    y_test, classifier.predict(X_test), pos_label=classifier.classes_[1]
)
recall = recall_score(
    y_test, classifier.predict(X_test), pos_label=classifier.classes_[1]
)

# %%
display = PrecisionRecallDisplay.from_estimator(hist_gradient_boosting, X_test, y_test)
display.ax_.plot(recall, precision, marker="o", label="Fixed threshold classifier")
display.ax_.legend()
_ = display.ax_.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curve")

# %%
classifier.set_params(threshold=0.1).fit(X_train, y_train)
precision = precision_score(
    y_test, classifier.predict(X_test), pos_label=classifier.classes_[1]
)
recall = recall_score(
    y_test, classifier.predict(X_test), pos_label=classifier.classes_[1]
)
display = PrecisionRecallDisplay.from_estimator(hist_gradient_boosting, X_test, y_test)
display.ax_.plot(recall, precision, marker="o", label="Fixed threshold classifier")
display.ax_.legend()
_ = display.ax_.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curve")

# %%
from sklearn.metrics import RocCurveDisplay

display = RocCurveDisplay.from_estimator(hist_gradient_boosting, X_test, y_test)
_ = display.ax_.set(
    xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC curve"
)
# %%
from sklearn.calibration import CalibrationDisplay

display = CalibrationDisplay.from_estimator(
    hist_gradient_boosting, X, y, n_bins=5, pos_label=hist_gradient_boosting.classes_[1]
)
_ = display.ax_.set(xlabel="Mean predicted probability", ylabel="Fraction of positives")

# %%
from sklearn.linear_model import LogisticRegression
from skrub import tabular_learner

logistic_regression = tabular_learner(LogisticRegression(C=1e-2)).fit(X_train, y_train)
logistic_regression

# %%
display = CalibrationDisplay.from_estimator(
    hist_gradient_boosting, X, y, n_bins=5, pos_label=hist_gradient_boosting.classes_[1]
)
CalibrationDisplay.from_estimator(
    logistic_regression,
    X,
    y,
    n_bins=5,
    pos_label=logistic_regression.classes_[1],
    ax=display.ax_,
)
_ = display.ax_.set(xlabel="Mean predicted probability", ylabel="Fraction of positives")

# %%
from sklearn.metrics import log_loss

log_loss_hgbdt = log_loss(y_test, hist_gradient_boosting.predict_proba(X_test))
log_loss_rf = log_loss(y_test, logistic_regression.predict_proba(X_test))

print(f"Log loss of the HistGradientBoostingClassifier: {log_loss_hgbdt:.2f}")
print(f"Log loss of the LogisticRegression: {log_loss_rf:.2f}")

# %%
from sklearn.metrics import brier_score_loss


brier_score_hgbdt = brier_score_loss(
    y_test,
    hist_gradient_boosting.predict_proba(X_test)[:, 1],
    pos_label=hist_gradient_boosting.classes_[1],
)
brier_score_rf = brier_score_loss(
    y_test,
    logistic_regression.predict_proba(X_test)[:, 1],
    pos_label=logistic_regression.classes_[1],
)

print(f"Brier score of the HistGradientBoostingClassifier: {brier_score_hgbdt:.2f}")
print(f"Brier score of the LogisticRegression: {brier_score_rf:.2f}")

# %%
display = PrecisionRecallDisplay.from_estimator(hist_gradient_boosting, X_test, y_test)
display.ax_.legend()
_ = display.ax_.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curve")

# %%
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import TunedThresholdClassifierCV,

scorer = make_scorer(f1_score, pos_label=hist_gradient_boosting.classes_[1])
tuned_threshold_classifier = TunedThresholdClassifierCV(
    hist_gradient_boosting,
    cv=3,
    scoring=scorer,
).fit(X_train, y_train)

# %%
tuned_threshold_classifier.best_threshold_

# %%
precision = precision_score(
    y_test,
    tuned_threshold_classifier.predict(X_test),
    pos_label=tuned_threshold_classifier.classes_[1],
)
recall = recall_score(
    y_test,
    tuned_threshold_classifier.predict(X_test),
    pos_label=tuned_threshold_classifier.classes_[1],
)
display = PrecisionRecallDisplay.from_estimator(hist_gradient_boosting, X_test, y_test)
display.ax_.plot(recall, precision, marker="o", label="Tuned threshold classifier")
display.ax_.legend()
_ = display.ax_.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curve")

# %%
import numpy as np


def max_recall_at_min_precision(y_true, y_pred, min_precision, pos_label):
    precision = precision_score(y_true.tolist(), y_pred, pos_label=pos_label)
    recall = recall_score(y_true.tolist(), y_pred, pos_label=pos_label)
    if precision < min_precision:
        return -np.inf
    return recall


scorer = make_scorer(
    max_recall_at_min_precision,
    min_precision=0.5,
    pos_label=hist_gradient_boosting.classes_[1],
)
tuned_threshold_classifier.set_params(scoring=scorer, store_cv_results=True).fit(
    X_train, y_train
)

# %%
tuned_threshold_classifier.best_threshold_

# %%
precision = precision_score(
    y_test,
    tuned_threshold_classifier.predict(X_test),
    pos_label=tuned_threshold_classifier.classes_[1],
)
recall = recall_score(
    y_test,
    tuned_threshold_classifier.predict(X_test),
    pos_label=tuned_threshold_classifier.classes_[1],
)
display = PrecisionRecallDisplay.from_estimator(hist_gradient_boosting, X_test, y_test)
display.ax_.plot(recall, precision, marker="o", label="Tuned threshold classifier")
display.ax_.legend()
_ = display.ax_.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curve")

# %%
