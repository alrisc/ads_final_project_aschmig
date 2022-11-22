import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from ipywidgets import interact
from time import time

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("ds_salaries.csv")

df_original = df.drop(['work_year','Unnamed: 0', 'salary', 'salary_currency', 'employee_residence','job_title','company_location'],axis=1)

df_original['work_year'] = df_original['work_year'].astype('object',copy=False)
df_original['remote_ratio'] = df_original['remote_ratio'].astype('object',copy=False)

X = df_original.drop('salary_in_usd',axis=1)
y = df_original['salary_in_usd']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

#section below creates which of the columns will be used as cat and num features
columns = X_train.columns.values

types = np.array([z for z in X_train.dtypes])

#This creates an array of the numerical features that also are not objects.
numerical = types != 'object'

num_features = columns[numerical].tolist()
cat_features = columns[~numerical].tolist() 

features = num_features + cat_features

#num pipeline filled with imputer mean
num_pipeline = Pipeline([
        #('imputer', SimpleImputer(strategy="mean")),
        ('std_scaler', StandardScaler()),
    ])

#cat pipeline filled values with none constant
cat_pipeline = Pipeline([
        #('imputer', SimpleImputer(strategy='constant', fill_value= "None")),
        ('ohe', OneHotEncoder(sparse=False, handle_unknown="ignore"))
    ])

#putting all cat and num together in this pipeline
data_pipeline = ColumnTransformer( transformers= [
        #(name, transformer,     columns)
        ("num_pipeline", num_pipeline, num_features),
        ("cat_pipeline", cat_pipeline, cat_features)],
         remainder='drop',
         n_jobs=-1
    )

#fitting train and valid set
data_pipeline.fit(X_train[features])
#data_pipeline.fit(X_valid[features])

#transforming the X train, valid, and test sets.
X_train_transformed = data_pipeline.transform(X_train[features])
#X_valid_transformed = data_pipeline.transform(X_valid[features])
X_test_transformed = data_pipeline.transform(X_test[features])


number_of_inputs = X_train_transformed.shape[1]

np.random.seed(42)
full_pipeline_with_predictor = Pipeline([
        ("preparation", data_pipeline),
        ("logistic", LogisticRegression(
            penalty='l1', 
            C = 1, 
            solver="saga"))
    ])

classifier = full_pipeline_with_predictor

classifier.fit(X_train, y_train)

#LRmodel = full_pipeline_with_predictor.fit(X_train, y_train)

import pickle

pickle.dump(classifier, open("model.pkl","wb"))