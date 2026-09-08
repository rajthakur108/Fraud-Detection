import mlflow
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from mlflow.tracking import MlflowClient
from prefect import flow, task
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier

TRACKING_URI = "http://127.0.0.1:5000"
registered_model_name = "fraud-detection-model"
mlflow.set_tracking_uri(uri=TRACKING_URI)
mlflow.set_experiment(experiment_name="fraud-detection")
client = MlflowClient(tracking_uri=TRACKING_URI)


@task(retries=3, retry_delay_seconds=2)
def read_dataframe(filename):
    """Function for reading the dataframe"""
    # print("Reading the Data")
    df = pd.read_csv(filename)
    df.drop_duplicates(inplace=True)
    X = df.drop(["Class"], axis=1)
    y = df["Class"]

    return X, y


@task
def split_data(X, y):
    """Function for splitting the data"""
    # print("Splitting the data into training,val,test")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.5, stratify=y_test, random_state=42
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


@task
def train_model(X_train, y_train):
    """Training script"""
    # print("Training the Model")
    model_uri = f"models:/{registered_model_name}@Champion"
    model = mlflow.sklearn.load_model(model_uri=model_uri)
    model = clone(
        model
    )  # Clone uses exact same hyperparameters that were used to train the model but does not attach any training details(basically trains the model from scratch)
    model.fit(X_train, y_train)
    # print("Training completed")
    return model


@task
def evaluate_model(model, X_val, y_val):
    """Function for evaluating the model"""
    # print("Evaluating the Model")
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]
    # print("Evaluation completed")
    return y_pred, y_proba


@task
def log_model(model, y_pred, y_proba, y_val):
    """Function for logging the model"""
    # print("Logging the trained Model")
    model_version = client.get_model_version_by_alias(
        name=registered_model_name, alias="Champion"
    )
    run_id = model_version.run_id
    info = client.get_run(run_id)
    model_name = info.data.params["model"]
    imbalance_method = info.data.params["imbalance_method"]
    params = info.data.params
    with mlflow.start_run() as run:
        pr_auc = average_precision_score(y_val, y_proba)
        mlflow.log_param("model", model_name)
        mlflow.log_param("imbalance_method", imbalance_method)
        mlflow.log_params(params)
        mlflow.log_metric("pr-auc", pr_auc)
        mlflow.log_metric("precision", precision_score(y_val, y_pred))
        mlflow.log_metric("recall", recall_score(y_val, y_pred))
        mlflow.log_metric("f1-score", f1_score(y_val, y_pred))
        mlflow.log_metric("accuracy_score", accuracy_score(y_pred, y_val))
        mlflow.sklearn.log_model(model, artifact_path="model")
        print("Congratulation! ☺️ Model Logged and Process completed")
        return run.info.run_id


@task
def register_model(run_id):
    model_uri = f"runs:/{run_id}/model"
    registered = mlflow.register_model(name=registered_model_name, model_uri=model_uri)
    return registered.version


@task
def promote_model(y_val, y_proba, version):
    pr_auc_for_latest_model = average_precision_score(y_val, y_proba)

    champion_model_version = client.get_model_version_by_alias(
        name=registered_model_name, alias="Champion"
    )
    run_id = champion_model_version.run_id
    info = client.get_run(run_id)
    pr_auc_for_champion_model = info.data.metrics["pr-auc"]

    if pr_auc_for_latest_model > pr_auc_for_champion_model:
        client.set_registered_model_alias(
            name=registered_model_name, alias="Champion", version=version
        )


@flow
def run():
    """Main flow"""
    X, y = read_dataframe("creditcard.csv")
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)
    model = train_model(X_train, y_train)
    y_pred, y_proba = evaluate_model(model, X_val, y_val)
    run_id = log_model(model, y_pred, y_proba, y_val)
    version = register_model(run_id)
    promote_model(y_val, y_proba, version)


if __name__ == "__main__":
    run()
