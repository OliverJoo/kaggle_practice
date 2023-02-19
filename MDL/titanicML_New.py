import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd

# to see all data
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 200)
pd.set_option('display.colheader_justify', 'left')

np.random.seed(0)

X, y = fetch_openml('titanic', version=1, as_frame=True, return_X_y=True)

number_features_median = ['age']
number_tranformer_median = Pipeline(steps=[('imputer_median', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])

number_features_constant = ['fare']
number_tranformer_constant = Pipeline(steps=[('imputer_constant', SimpleImputer(strategy='constant', fill_value=0)), ('scaler', StandardScaler())])

categorical_features = ['embarked', 'sex', 'pclass']
categorical_tranformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(transformers=[('number_median', number_tranformer_median, number_features_median),
                                               ('number_constant', number_tranformer_constant, number_features_constant),
                                               ('category', categorical_tranformer, categorical_features)])

lr_clf = Pipeline(steps=[('prepro', preprocessor), ('classifier', LogisticRegression(solver='liblinear', warm_start=True, tol=0.01, random_state=20))])
dt_clf = Pipeline(steps=[('prepro', preprocessor), ('classifier', DecisionTreeClassifier(random_state=20))])
rf_clf = Pipeline(steps=[('prepro', preprocessor), ('classifier', RandomForestClassifier(random_state=20, n_estimators=300, max_depth=23))])
lgb_clf = Pipeline(steps=[('prepro', preprocessor), ('classifier', lgb.LGBMClassifier(random_state=20, n_estimators=300))])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=20)

lr_clf.fit(X_train, y_train)
print(f'model score: {lr_clf.score(X_test, y_test)}')

dt_clf.fit(X_train, y_train)
print(f'model score: {dt_clf.score(X_test, y_test)}')

rf_clf.fit(X_train, y_train)
print(f'model score: {rf_clf.score(X_test, y_test)}')

lgb_clf.fit(X_train, y_train)
print(f'model score: {lgb_clf.score(X_test, y_test)}')