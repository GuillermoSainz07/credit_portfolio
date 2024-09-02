import pandas as pd
import numpy as np

import mlflow
import xgboost as xgb
from hyperparams import params
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import TargetEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss, accuracy_score, f1_score

import os

from dotenv import load_dotenv

load_dotenv()

data = pd.read_csv('credit_risk_dataset.csv')
data = data.drop(index=data.query('person_emp_length > 100').index)

X_cat = data.drop(columns=['loan_status','cb_person_default_on_file',
                           'person_home_ownership']).select_dtypes(include=['object'])

X_num = data.drop(columns='loan_status').select_dtypes(include=['float','int'])
y = data.loan_status.to_numpy().flatten()

X = X_num.join(X_cat)

X_train, X_test, y_train, y_test = train_test_split(X,y, stratify=y)

X_train,X_val, y_train, y_val = train_test_split(X_train,y_train,stratify=y_train, train_size=0.70)

col_trans = ColumnTransformer([("cat_preprocess", TargetEncoder(), ["loan_grade"]),
                                ("cat_preprocess_2", TargetEncoder(), ["loan_intent"])],
                                remainder='passthrough').set_output(transform='pandas')

col_trans.fit_transform(X_train,y_train)

X_train = col_trans.transform(X_train)
X_val = col_trans.transform(X_val)
X_test = col_trans.transform(X_test)


TRACKING_SERVER_HOST = "ec2-3-133-151-150.us-east-2.compute.amazonaws.com"

mlflow.set_tracking_uri(f'http://{TRACKING_SERVER_HOST}:5000')
print(f'Tracking Server URI: {mlflow.get_tracking_uri()}')
mlflow.set_experiment('xgb_pipeline_credit_model')


with mlflow.start_run(run_name='XGB Credit Classifier'):
    xgb_clf = xgb.XGBClassifier(**params,
                               early_stopping_rounds=20)

    xgb_clf.fit(X_train,y_train,
                eval_set=[(X_val, y_val)],
                verbose=5)
    
    pipe = Pipeline([('transformer',col_trans),
                 ('xgb_model',xgb_clf)])
    
    y_pred = xgb_clf.predict(X_test)
    y_pred_probabilities = xgb_clf.predict_proba(X_test)[:, 1]

    mlflow.log_params(xgb_clf.get_params())
    mlflow.log_metric('log_loss', log_loss(y_test, y_pred_probabilities))
    mlflow.log_metric('accuracy', accuracy_score(y_test, y_pred))
    mlflow.log_metric('f1_score', f1_score(y_test, y_pred))
    mlflow.xgboost.log_model(xgb_clf,'xgb_model')
    mlflow.sklearn.log_model(pipe, "xgb_pipeline")

    # Registrar el modelo
    mlflow.register_model("xgb_pipeline",
                          "XGB Credit Classifier")
    