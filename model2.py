#Here is where we would place all our machine learning model code.

#%matplotlib inline
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

import seaborn as sns; sns.set()

import pickle











df = pd.read_csv("ds_salaries.csv")

df_original = df.drop(['Unnamed: 0', 'salary', 'salary_currency', 'employee_residence'],axis=1)

df = df.drop('Unnamed: 0',axis = 1)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

col_list = ['work_year',
            'experience_level',
            'employment_type',
            'job_title',
            'salary_currency',
            'employee_residence',
            'remote_ratio',
            'company_location',
            'company_size']

for column in col_list:
    df[column] = le.fit_transform(df[column])






df_original['work_year'] = df_original['work_year'].astype('object',copy=False)
df_original['remote_ratio'] = df_original['remote_ratio'].astype('object',copy=False)

dummies = df_original.drop('salary_in_usd',axis=1)
dum_imp = df_original.drop('salary_in_usd',axis=1)

#Then I impute the missing values, and refit them.
#myImputer = SimpleImputer(strategy = 'most_frequent')
#dum_imp = pd.DataFrame(myImputer.fit_transform(dummies))

dum_imp.columns = dummies.columns
dum_cols = dum_imp.columns.tolist()

dum_imp = dum_imp.join(pd.get_dummies(dum_imp[dum_cols], prefix = dum_cols ))
dum_imp = dum_imp.drop(dum_cols, axis = 1)

frames = [df_original['salary_in_usd'], dum_imp]
df_original = pd.concat(frames, axis = 1)

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

print(X.head())

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



#Experiment Log: modified for Continuous Target Variables
try:
    expLog = pd.DataFrame(columns=["exp_name", 
                                   "Train MSE", 
                                   "Test  MSE",
                                   "Train RMSE", 
                                   "Test  RMSE",
                                   "Train MAE", 
                                   "Test  MAE",
                                   "Train r-squared",
                                   "Test r-squared",
                                  ])
except NameError:
    expLog = pd.DataFrame(columns=["exp_name", 
                                   "Train MSE", 
                                   "Test  MSE",
                                   "Train RMSE", 
                                   "Test  RMSE",
                                   "Train MAE", 
                                   "Test  MAE",
                                   "Train r-squared",
                                   "Test r-squared",
                                  ])


#These are the various models that will run along with the experiment log.

experiment_name = f'_penalty=l1_C=1_solver=saga'

np.random.seed(42)
full_pipeline_with_predictor = Pipeline([
        ("preparation", data_pipeline),
        ("logistic", LogisticRegression(
            penalty='l1', 
            C = 1, 
            solver="saga"))
    ])
LRmodel = full_pipeline_with_predictor.fit(X_train, y_train)

exp_name = f"Baseline_LogReg_{experiment_name}"
expLog.loc[len(expLog)] = [f"{exp_name}"] + list(np.round(
               [mean_squared_error(y_train, LRmodel.predict(X_train)), 
                mean_squared_error(y_test, LRmodel.predict(X_test)),
                np.sqrt(mean_squared_error(y_train, LRmodel.predict(X_train))),
                np.sqrt(mean_squared_error(y_test, LRmodel.predict(X_test))),
                mean_absolute_error(y_train, LRmodel.predict(X_train)),
                mean_absolute_error(y_test, LRmodel.predict(X_test)),
                r2_score(y_train, LRmodel.predict(X_train)),
                r2_score(y_test, LRmodel.predict(X_test)),
               ],
                
                4)) 

experiment_name = f'_penalty=l1_C=10_solver=saga'

np.random.seed(42)
full_pipeline_with_predictor = Pipeline([
        ("preparation", data_pipeline),
        ("logistic", LogisticRegression(
            penalty='l1', 
            C = 10, 
            solver="saga"))
    ])
LRmodel = full_pipeline_with_predictor.fit(X_train, y_train)

exp_name = f"Baseline_LogReg_{experiment_name}"
expLog.loc[len(expLog)] = [f"{exp_name}"] + list(np.round(
               [mean_squared_error(y_train, LRmodel.predict(X_train)), 
                mean_squared_error(y_test, LRmodel.predict(X_test)),
                np.sqrt(mean_squared_error(y_train, LRmodel.predict(X_train))),
                np.sqrt(mean_squared_error(y_test, LRmodel.predict(X_test))),
                mean_absolute_error(y_train, LRmodel.predict(X_train)),
                mean_absolute_error(y_test, LRmodel.predict(X_test)),
                r2_score(y_train, LRmodel.predict(X_train)),
                r2_score(y_test, LRmodel.predict(X_test)),
               ],
                
                4)) 

experiment_name = f'_penalty=l1_C=100_solver=saga'

np.random.seed(42)
full_pipeline_with_predictor = Pipeline([
        ("preparation", data_pipeline),
        ("logistic", LogisticRegression(
            penalty='l1', 
            C = 100, 
            solver="saga"))
    ])
LRmodel = full_pipeline_with_predictor.fit(X_train, y_train)

exp_name = f"Baseline_LogReg_{experiment_name}"
expLog.loc[len(expLog)] = [f"{exp_name}"] + list(np.round(
               [mean_squared_error(y_train, LRmodel.predict(X_train)), 
                mean_squared_error(y_test, LRmodel.predict(X_test)),
                np.sqrt(mean_squared_error(y_train, LRmodel.predict(X_train))),
                np.sqrt(mean_squared_error(y_test, LRmodel.predict(X_test))),
                mean_absolute_error(y_train, LRmodel.predict(X_train)),
                mean_absolute_error(y_test, LRmodel.predict(X_test)),
                r2_score(y_train, LRmodel.predict(X_train)),
                r2_score(y_test, LRmodel.predict(X_test)),
               ],
                
                4)) 

print(expLog)




#Below is the template version of what you need to do to apply a model in the web app properly.
#You will likely need to swap out the model to be a more appropriate model.

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

classifier = LogisticRegression(
            penalty='l1', 
            C = 100, 
            solver="saga")

classifier.fit(X_train,y_train)

#Then we make a pickle file for the trained model.

pickle.dump(classifier, open("model.pkl","wb"))