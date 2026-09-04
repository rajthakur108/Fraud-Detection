import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler,StandardScaler
import mlflow
from prefect import flow,task
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler,SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,average_precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from mlflow.tracking import MlflowClient

TRACKING_URI = "http://127.0.0.1:5000"
registered_model_name = "fraud-detection-model"
mlflow.set_tracking_uri(uri=TRACKING_URI)
mlflow.set_experiment(experiment_name="fraud-detection")
client = MlflowClient(tracking_uri = TRACKING_URI)

@task(retries=3,retry_delay_seconds=2)
def read_dataframe(filename):
    """Function for reading the dataframe"""
    print("Reading the Data")
    df = pd.read_csv(filename)
    df.drop_duplicates(inplace=True)
    X = df.drop(['Class'],axis = 1)
    y = df['Class']

    return X,y

@task
def split_data(X,y):
    """Function for splitting the data"""
    print("Splitting the data into training,val,test")
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,stratify=y,random_state=42)
    X_val,X_test,y_val,y_test = train_test_split(X_test,y_test,test_size=0.5,stratify=y_test,random_state=42)
    return X_train,y_train,X_val,y_val,X_test,y_test

@task
def feature_engineering(X,amount_scaler,time_scaler,pca_scaler):
    """Function for transforming the features"""
    print(f"Transforming the features")
    if amount_scaler is None:
        amount_scaler = RobustScaler()
        time_scaler = RobustScaler()
        pca_scaler = RobustScaler()
        X['Amount'] = np.log1p(X['Amount'])
        X['Amount'] = amount_scaler.fit_transform(X[['Amount']])
        X['Time'] = time_scaler.fit_transform(X[['Time']])
        v_cols = [f"V{i}" for i in range(1,29)]
        X[v_cols] = pca_scaler.fit_transform(X[v_cols])
        return amount_scaler,time_scaler,pca_scaler

    else:
        X['Amount'] = np.log1p(X['Amount'])
        X['Amount'] = amount_scaler.transform(X[['Amount']])
        X['Time'] = time_scaler.transform(X[['Time']])
        v_cols = [f"V{i}" for i in range(1,29)]
        X[v_cols] = pca_scaler.transform(X[v_cols])

def resample_data(X_train,y_train,imbalance_method):
    """This function resamples the data according to the provided imbalance method"""
    if imbalance_method == 'undersample':
        sampler = RandomUnderSampler(random_state=42)
    elif imbalance_method == 'oversample':
        sampler = RandomOverSampler(random_state=42)
    else:
        sampler = SMOTE(random_state=42)

    X_train_resampled,y_train_resampled = sampler.fit_resample(X_train,y_train)
    return X_train_resampled,y_train_resampled

@task
def train_model(X_train,y_train):
    """Training script"""
    print("Training the Model")
    model = client.get_model_version_by_alias(name = registered_model_name,alias = "Champion")
    run_id = model.run_id
    info = client.get_run(run_id)

    imbalance_method = info.data.params["imbalance_method"]
    print("Resampling the data")
    X_train_resampled,y_train_resampled = resample_data(X_train,y_train,imbalance_method)

    model = info.data.params["model"]

    if model == 'Logistic Regression':
        params = {"C":float(info.data.params["C"])}
        classifier = LogisticRegression(**params)
    elif model == 'Decision Tree':
        params = {
            "max_depth":int(info.data.params["max_depth"]),
            "min_samples_leaf":int(info.data.params["min_samples_leaf"]),
            "min_samples_split":int(info.data.params["min_samples_split"]),
            }
        classifier = DecisionTreeClassifier(**params)
    else:
        params = {
            "max_depth":int(info.data.params["max_depth"]),
            "n_estimators":int(info.data.params["n_estimators"]),
            "min_samples_leaf":int(info.data.params["min_samples_leaf"]),
            "min_samples_split":int(info.data.params["min_samples_split"]),
        }
        classifier = RandomForestClassifier(**params)

    classifier.fit(X_train_resampled,y_train_resampled)
    print("Training completed")
    return classifier,params,imbalance_method,model

@task
def evaluate_model(classifier,X_val,y_val):
    """Function for evaluating the model"""
    print("Evaluating the Model")
    y_pred = classifier.predict(X_val)
    y_proba = classifier.predict_proba(X_val)[:,1]
    print("Evaluation completed")
    return y_pred,y_proba

@task
def log_model(classifier,y_pred,y_proba,y_val,params,amount_scaler,time_scaler,pca_scaler,imbalance_method,model):
    """Function for logging the model"""
    print("Logging the trained Model")
    with mlflow.start_run():
        pr_auc = average_precision_score(y_val,y_proba)
        mlflow.log_param("model",model)
        mlflow.log_param("imbalance_method",imbalance_method)
        mlflow.log_params(params)
        mlflow.log_metric("pr-auc",pr_auc)
        mlflow.log_metric("precision",precision_score(y_val,y_pred))
        mlflow.log_metric("recall",recall_score(y_val,y_pred))
        mlflow.log_metric("f1-score",f1_score(y_val,y_pred))
        mlflow.log_metric("accuracy_score",accuracy_score(y_pred,y_val))
        mlflow.log_artifact()
        mlflow.sklearn.log_model(classifier,artifact_path="model")
    print("Congratulation! ☺️ Model Logged and Process completed")

@flow
def run():
    """Main flow"""
    X,y = read_dataframe('creditcard.csv')
    X_train,y_train,X_val,y_val,X_test,y_test = split_data(X,y)
    amount_scaler,time_scaler,pca_scaler = feature_engineering(X_train,None,None,None)
    feature_engineering(X_val,amount_scaler,time_scaler,pca_scaler)
    feature_engineering(X_test,amount_scaler,time_scaler,pca_scaler)

    classifier,params,imbalance_method,model = train_model(X_train,y_train)
    y_pred,y_proba = evaluate_model(classifier,X_val,y_val)
    log_model(classifier,y_pred,y_proba,y_val,params,amount_scaler,time_scaler,pca_scaler,imbalance_method,model)
    

if __name__ == "__main__":
    run()