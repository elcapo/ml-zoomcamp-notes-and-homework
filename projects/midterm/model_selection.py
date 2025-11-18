import pandas as pd
import matplotlib.pylab as plt
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb

def search_random_forest(X_train: pd.DataFrame, y_train: pd.DataFrame, param_distributions: dict) -> RandomizedSearchCV:
    random_forest = RandomForestClassifier(random_state=1)
    random_forest_search = RandomizedSearchCV(random_forest, param_distributions, scoring="roc_auc", n_jobs=8)
    random_forest_search.fit(X_train, y_train)

    return random_forest_search

def search_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    param_distributions: dict
) -> RandomizedSearchCV:
    booster = xgb.XGBClassifier(
        tree_method="hist",
        early_stopping_rounds=2,
        eval_metric=roc_auc_score,
        random_state=1,
        objective="binary:logistic",
        nthread=8,
    )

    booster_search = RandomizedSearchCV(booster, param_distributions)
    booster_search.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    return booster_search

def train_random_forest(X: pd.DataFrame, y: pd.DataFrame, param_distributions: dict) -> RandomForestClassifier:
    random_forest = RandomForestClassifier(random_state=1, **param_distributions)
    random_forest.fit(X, y)

    return random_forest

def train_booster(X: pd.DataFrame, y: pd.DataFrame, param_distributions: dict) -> xgb.XGBClassifier:
    booster = xgb.XGBClassifier(random_state=1, **param_distributions)
    booster.fit(X, y)

    return booster

def eval_model(X_val: pd.DataFrame, y_val: pd.DataFrame, model: ClassifierMixin):
    y_pred = model.predict(X_val)

    return roc_auc_score(y_val, y_pred)

def plot_experiments(search: RandomizedSearchCV):
    fit_times = search.cv_results_["mean_fit_time"]
    scores = search.cv_results_["mean_test_score"]

    fig, ax_scores = plt.subplots(figsize=(12, 6))
    ax_fit_times = ax_scores.twinx()

    colors = ["royalblue" for _ in range(len(search.cv_results_))]
    colors[search.best_index_] = "forestgreen"

    ax_scores.bar(x=range(len(scores)), height=scores, color=colors)
    ax_scores.set_xlabel("Experiment")
    ax_scores.set_ylabel("Mean test score")

    ax_fit_times.plot(range(len(scores)), fit_times, color="red")
    ax_scores.set_xticks(range(len(scores)))
    ax_fit_times.set_ylabel("Mean fit time")

    plt.title("Experiment analysis")
    plt.show()

def plot_random_forest_parameters(search: RandomizedSearchCV):
    _, axis = plt.subplots(1, 3, figsize=(16, 3))

    colors = ["royalblue" for _ in range(len(search.cv_results_))]
    colors[search.best_index_] = "forestgreen"

    n_estimators = [param["n_estimators"] if param["n_estimators"] != None else 0 for param in search.cv_results_["params"]]
    ax_estimators = axis[0]
    ax_estimators.bar(x=range(len(n_estimators)), height=n_estimators, color=colors)
    ax_estimators.set_xticks(range(len(n_estimators)))
    ax_estimators.set_xlabel("Experiment")
    ax_estimators.set_title("Number of estimators")

    min_samples_leaf = [param["min_samples_leaf"] for param in search.cv_results_["params"]]
    ax_min_samples_leaf = axis[1]
    ax_min_samples_leaf.bar(x=range(len(min_samples_leaf)), height=min_samples_leaf, color=colors)
    ax_min_samples_leaf.set_xticks(range(len(min_samples_leaf)))
    ax_min_samples_leaf.set_xlabel("Experiment")
    ax_min_samples_leaf.set_title("Mimimum samples per leaf")

    max_depth = [param["max_depth"] if param["max_depth"] != None else 0 for param in search.cv_results_["params"]]
    ax_max_depth = axis[2]
    ax_max_depth.bar(x=range(len(max_depth)), height=max_depth, color=colors)
    ax_max_depth.set_xticks(range(len(max_depth)))
    ax_max_depth.set_xlabel("Experiment")
    ax_max_depth.set_title("Maximum depth")

    plt.show()

def plot_xgboost_parameters(search: RandomizedSearchCV):
    _, axis = plt.subplots(1, 3, figsize=(16, 3))

    colors = ["royalblue" for _ in range(len(search.cv_results_))]
    colors[search.best_index_] = "forestgreen"

    eta = [param["eta"] for param in search.cv_results_["params"]]
    ax_eta = axis[0]
    ax_eta.bar(x=range(len(eta)), height=eta, color=colors)
    ax_eta.set_xticks(range(len(eta)))
    ax_eta.set_xlabel("Experiment")
    ax_eta.set_title("Eta")

    max_depth = [param["max_depth"] if param["max_depth"] != None else 0 for param in search.cv_results_["params"]]
    ax_max_depth = axis[1]
    ax_max_depth.bar(x=range(len(max_depth)), height=max_depth, color=colors)
    ax_max_depth.set_xticks(range(len(max_depth)))
    ax_max_depth.set_xlabel("Experiment")
    ax_max_depth.set_title("Maximum depth")

    min_child_weight = [param["min_child_weight"] for param in search.cv_results_["params"]]
    ax_min_child_weight = axis[2]
    ax_min_child_weight.bar(x=range(len(min_child_weight)), height=min_child_weight, color=colors)
    ax_min_child_weight.set_xticks(range(len(min_child_weight)))
    ax_min_child_weight.set_xlabel("Experiment")
    ax_min_child_weight.set_title("Mimimum child weight")

    plt.show()
