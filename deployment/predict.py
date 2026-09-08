import os

import joblib
import mlflow
import pandas as pd
from flask import Flask, jsonify, request

model = joblib.load("fraud_model.joblib")


def make_prediction(model, transaction):
    transaction_df = pd.DataFrame([transaction])
    y_pred = model.predict(transaction_df)
    y_proba = model.predict_proba(transaction_df)[:, 1]
    return int(y_pred[0]), float(y_proba[0])


app = Flask("fraud-detection")


@app.route("/predict", methods=["POST"])
def predict():
    transaction = request.get_json()
    y_pred, y_proba = make_prediction(model, transaction)
    prediction = {"fraud": y_pred, "probability_of_fraud": y_proba}

    return jsonify(prediction)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
