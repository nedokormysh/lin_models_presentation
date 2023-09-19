import pandas as pd
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import streamlit as st
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.metrics import classification_report
import joblib
from io import BytesIO
import requests


@st.cache_data
def load_data(data, test_size=0.2, feat=False):
    # all_data = pd.read_csv('https://raw.githubusercontent.com/nedokormysh/lin_models_presentation/main/all_data.csv')
    all_data = data
    targets = ['TARGET']
    features2drop = ['WORK_TIME', 'FACT_ADDRESS_PROVINCE', 'REG_ADDRESS_PROVINCE']
    filtered_features = [i for i in all_data.columns if (i not in targets and i not in features2drop)]

    continuous_features = ['CREDIT', 'FST_PAYMENT', 'AGE', 'CHILD_TOTAL', 'DEPENDANTS',
                           'OWN_AUTO', 'PERSONAL_INCOME', 'WORK_TIME', 'CLOSED_LOANS', 'LOAN_AMOUNT',
                           'WORK_TIME_IN_YEARS']
    categorical_features = [i for i in all_data.columns if
                            i not in continuous_features and (i not in targets and i not in features2drop)]

    num_features = [i for i in filtered_features if i not in categorical_features]

    for col in categorical_features:
        all_data[col] = all_data[col].astype("category")

    X = all_data[filtered_features].drop(targets, axis=1, errors="ignore")
    y = all_data['TARGET']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    if feat:
        return X_train, X_test, y_train, y_test, num_features, categorical_features
    else:
        return X_train, X_test, y_train, y_test

@st.cache_data
def load_model(path='https://github.com/nedokormysh/lin_models_presentation/raw/model_streamlit/model.pickle'):
    model = joblib.load(BytesIO(requests.get(path).content))

    return model

def top_features(model, num=6):
    coef_table = pd.DataFrame(
        {'weights': list(abs(model[-1].coef_[0])), 'features': list(model[:-1].get_feature_names_out())})

    return coef_table.sort_values(by='weights', ascending=False).head(num)

def class_report(model, X_test, y_test, threshold=0.5):
    probs = model.predict_proba(X_test)
    probs_churn = probs[:, 1]
    classes = probs_churn > threshold

    recall = recall_score(y_test, classes)
    precision = precision_score(y_test, classes)
    f1 = f1_score(y_test, classes)

    return recall, precision, f1

def single_prediction(model, X_test, id=1, threshold=0.5):
    name = X_test.iloc[id].name
    prob = model.predict_proba(X_test[X_test.index == name])[:, 1]
    predict_sngl = prob > threshold

    return prob, int(predict_sngl)

def pipeline(num_features, categorical_features):
    model = LogisticRegression(class_weight='balanced')

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")),
               ("scaler", StandardScaler())])
    categorical_transformer = Pipeline(
        steps=[("encoder", OneHotEncoder(handle_unknown="ignore")),
               #  ("selector", SelectPercentile(chi2, percentile=50)),
               ])
    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, num_features),
                      ("cat", categorical_transformer, categorical_features)])

    pipe = Pipeline([('feature_preprocessor', preprocessor),
                     ('model', model)])

    return pipe