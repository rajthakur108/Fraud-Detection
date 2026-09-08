import datetime
import logging
import time

import joblib
import pandas as pd
import psycopg
from evidently import ColumnMapping
from evidently.metrics import (ColumnDriftMetric, DatasetDriftMetric,
                               DatasetMissingValuesMetric)
from evidently.report import Report

SEND_TIMEOUT = 10
BATCH_SIZE = 5000
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
)
begin = datetime.datetime(2026, 9, 3, 0, 0)
reference_data = pd.read_csv("./data/reference_data.csv")
current_data = pd.read_csv("./data/current_data.csv")
model = joblib.load("../deployment/fraud_model.joblib")

column_mapping = ColumnMapping(target=None, prediction="prediction")

report = Report(
    metrics=[
        ColumnDriftMetric(column_name="prediction"),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric(),
    ]
)

create_table_statement = """
drop table if exists fraud_metrics;
create table fraud_metrics(
    timestamp timestamp,
    prediction_drift float,
    num_drifted_columns integer,
    share_missing_values float,
    fraud_prediction_rate float,
    fraud_prediction_proba_avg float
)
"""


def prep_db():
    """Function for creating the database and table inside the database"""
    with psycopg.connect(
        "host =localhost port=5432 user=postgres password=example", autocommit=True
    ) as conn:
        res = conn.execute("select 1 from pg_database where datname ='test'")
        if len(res.fetchall()) == 0:  # No database named test
            conn.execute("create database test;")
        with psycopg.connect(
            "host=localhost port=5432 dbname = test user=postgres password=example"
        ) as conn:
            conn.execute(create_table_statement)


def calculate_metric_postgres(curr, i, start_day):
    """Function for calculating thr prediction on a single row of the
    current data(simulated as single day) and then logging the resuts in the table(fraud_metrics)
    """
    data = current_data.iloc[i : (i + BATCH_SIZE)]
    data["prediction"] = model.predict(data)
    data["fraud_probability"] = model.predict_proba(data)[:, 1]

    report.run(
        reference_data=reference_data, current_data=data, column_mapping=column_mapping
    )
    results = report.as_dict()

    drift_score = results["metrics"][0]["result"]["drift_score"]
    num_drifted_columns = results["metrics"][1]["result"]["number_of_drifted_columns"]
    share_of_missing_values = results["metrics"][2]["result"]["current"][
        "share_of_missing_values"
    ]
    fraud_prediction_rate = data["prediction"].mean()
    fraud_prediction_proba_avg = data["fraud_probability"].mean()

    curr.execute(
        "insert into fraud_metrics (timestamp,prediction_drift,num_drifted_columns,share_missing_values,fraud_prediction_rate,fraud_prediction_proba_avg) values(%s, %s, %s, %s, %s, %s)",
        (
            begin + datetime.timedelta(start_day),
            drift_score,
            num_drifted_columns,
            share_of_missing_values,
            fraud_prediction_rate,
            fraud_prediction_proba_avg,
        ),
    )


def run():
    start_day = -1
    prep_db()

    last_send = datetime.datetime.now()
    current_data_shape = current_data.shape[0]

    with psycopg.connect(
        "host = localhost port=5432 dbname = test user=postgres password=example",
        autocommit=True,
    ) as conn:
        for i in range(0, current_data_shape, BATCH_SIZE):
            with conn.cursor() as curr:
                start_day += 1
                calculate_metric_postgres(curr, i, start_day)

            new_send = datetime.datetime.now()
            time_elapsed = (new_send - last_send).total_seconds()
            if time_elapsed < SEND_TIMEOUT:
                time.sleep(SEND_TIMEOUT - time_elapsed)

            last_send = last_send + datetime.timedelta(seconds=10)
            logging.info("data sent")


if __name__ == "__main__":
    run()
