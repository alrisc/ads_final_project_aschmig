from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ipywidgets import interact
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import \
    train_test_split  # sklearn.cross_validation in old versions
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets

import seaborn as sns; sns.set()

import pickle

df = datasets.load_iris()

X = df.data
y = df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit(X_test)

classifier = RandomForestClassifier()

classifier.fit(X_train,y_train)

pickle.dump(classifier, open("model.pkl","wb"))