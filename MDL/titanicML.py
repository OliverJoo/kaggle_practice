import numpy as np
import pandas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# to see all data
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 200)
pd.set_option('display.colheader_justify', 'left')

titanic_df = pd.read_csv('./Data/train.csv')


# print(titanic_df.describe())


def prepare_df_chk(df: pandas.DataFrame):
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['Cabin'].fillna('N', inplace=True)
    df['Embarked'].fillna('N', inplace=True)
    df['Cabin'] = df['Cabin'].str[:1]
    return df


titanic_df = prepare_df_chk(titanic_df)


# sns.barplot(x='Sex', y='Survived', data=titanic_df)
# sns.barplot(x='Pclass', y='Survived', hue='Sex', data=titanic_df)
# plt.show()


# categorization by age
def get_category(age):
    cat = ''
    if age <= -1:
        cat = 'Unknown'
    elif age <= 5:
        cat = 'Baby'
    elif age <= 12:
        cat = 'Child'
    elif age <= 18:
        cat = 'Teenager'
    elif age <= 60:
        cat = 'Adult'
    else:
        cat = 'Elderly'

    return cat


plt.figure(figsize=(10, 6))

# X-axis ordering
group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Adult', 'Elderly']

titanic_df['Age_cat'] = titanic_df['Age'].apply(lambda x: get_category(x))
sns.barplot(x='Age_cat', y='Survived', hue='Sex', data=titanic_df, order=group_names)
titanic_df.drop('Age_cat', axis=1, inplace=True)

from sklearn.preprocessing import LabelEncoder


# Filling Null Parameters
def fillna(df):
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['Cabin'].fillna('N', inplace=True)
    df['Embarked'].fillna('N', inplace=True)
    df['Fare'].fillna(0, inplace=True)
    return df


# remove non-meaningful columns
def drop_features(df):
    df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
    return df


# Label Encoding
def format_features(df):
    df['Cabin'] = df['Cabin'].str[:1]
    features = ['Cabin', 'Sex', 'Embarked']
    for feature in features:
        le = LabelEncoder()
        le = le.fit(df[feature])
        df[feature] = le.transform(df[feature])
    return df


# preprocessing
def transform_features(df):
    df = fillna(df)
    df = drop_features(df)
    df = format_features(df)
    return df


# reloading original dataset to get feature and label dataset separately
titanic_df = pd.read_csv('./Data/train.csv')
test_titanic_df = pd.read_csv('./Data/test.csv')

X_train_df = transform_features(titanic_df)

X_test = transform_features(test_titanic_df)
X_train = X_train_df.drop('Survived', axis=1)
Y_train = X_train_df['Survived']

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

validation_titanic_df = pd.read_csv('./Data/gender_submission.csv')
# create DecisionTree, RandomForest, LogisticRegression Classifier
dt_clf = DecisionTreeClassifier(random_state=20)
rf_clf = RandomForestClassifier(random_state=20, n_estimators=300, max_depth=23)
lr_clf = LogisticRegression(solver='liblinear', warm_start=True, tol=0.01, random_state=20)
lgb_clf = lgb.LGBMClassifier(random_state=20, n_estimators=300)

# DecisionTreeClassifier learn/predict/evaluation
dt_clf.fit(X_train, Y_train)
dt_pred = dt_clf.predict(X_test)
print(f'DecisionTreeClassifier score: {dt_clf.score(X_train, Y_train):.4f}')  # \n{dt_pred}')
print(f'DecisionTreeClassifier MAE : {mean_absolute_error(validation_titanic_df["Survived"], dt_pred)}')

# RandomForestClassifier learn/predict/evaluation
rf_clf.fit(X_train, Y_train)
rf_pred = rf_clf.predict(X_test)
print(f'RandomForestClassifier score: {rf_clf.score(X_train, Y_train):.4f}')  # \n{rf_pred}')
print(f'RandomForestClassifier MAE : {mean_absolute_error(validation_titanic_df["Survived"], rf_pred)}')

# LogisticRegression learn/predict/evaluation
lr_clf.fit(X_train, Y_train)
lr_pred = lr_clf.predict(X_test)
print(f'LogisticRegression score: {lr_clf.score(X_train, Y_train):.4f}')  # \n{lr_pred}')
print(f'LogisticRegression MAE : {mean_absolute_error(validation_titanic_df["Survived"], lr_pred)}')

lgb_clf.fit(X_train, Y_train)
lgb_pred = lgb_clf.predict(X_test)
print(f'LGBMClassifier score: {lgb_clf.score(X_train, Y_train):.4f}')  # \n{lr_pred}')
print(f'LGBMClassifier MAE : {mean_absolute_error(validation_titanic_df["Survived"], lgb_pred)}')

# submission = pd.DataFrame({"PassengerId": pd.read_csv('./Data/test.csv')["PassengerId"], "Survived": rf_pred})
# submission.to_csv('submission.csv', index=False)

# from sklearn.model_selection import GridSearchCV
#
# try:
#     parameters = {'max_depth': [2, 3, 4, 5, 6, 7], 'min_samples_split': [2, 3, 4, 5],
#                   'min_samples_leaf': [1, 3, 5, 7, 9]}
#
#     # DecisionTreeClassifier
#     grid_dtlf = GridSearchCV(dt_clf, param_grid=parameters, scoring='accuracy', cv=5)
#     grid_dtlf.fit(X_train, Y_train)
#
#     print('\nDecisionTree GridSearchCV Optimal hyperparameter :', grid_dtlf.best_params_)
#     print(f'DecisionTree GridSearchCV best accuracy: {grid_dtlf.best_score_:.4f}')
#     best_dtlf = grid_dtlf.best_estimator_
#     # predict & evaluation by  best estimator(optimal hyperparameter of GridSearchCV)
#     dt_predictions = best_dtlf.predict(X_test)
#     print(f'DecisionTreeClassifier accuracy : {grid_dtlf.score(X_train, Y_train):.4f}')
#
#     # RandomForest
#     grid_rflf = GridSearchCV(rf_clf, param_grid=parameters, scoring='accuracy', cv=5)
#     grid_rflf.fit(X_train, Y_train)
#
#     print('\nRandomForest GridSearchCV Optimal hyperparameter :', grid_rflf.best_params_)
#     print(f'RandomForest GridSearchCV best accuracy: {grid_rflf.best_score_:.4f}')
#     best_dclf = grid_rflf.best_estimator_
#     # predict & evaluation by  best estimator(optimal hyperparameter of GridSearchCV)
#     rfpredictions = best_dclf.predict(X_test)
#     print(f'RandomForest accuracy : {grid_dtlf.score(X_train, Y_train):.4f}')
#
# except Exception as e:
#     print(e)

# plt.show()
