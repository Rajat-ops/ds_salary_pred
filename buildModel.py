import numpy as np
import pandas as pd 
import statsmodels.api as sm

df = pd.read_csv("salary_data_cleaned.csv")

df_models = df[['Rating', 'Size', 'Type of ownership', 'Industry', 'Sector', 'Revenue', 'hourly', 'job_state', 'age', 'aws', 'excel','avg_salary', 'python_yn','spark']]

df_dum = pd.get_dummies(df_models)

from sklearn.model_selection import train_test_split

X = df_dum.drop('avg_salary', axis=1)
y = df_dum.avg_salary.values  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

"""
// using OLS method

X_sum = X = sm.add_constant(X)

model = sm.OLS(y, X_sum)

print(model.fit().summary())

"""

"""
//using LinearRegression

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score

lr = LinearRegression()
lr.fit(X_train, y_train)

print(cross_val_score(lr, X_train, y_train, scoring = "neg_mean_absolute_error", cv=3))

lr_lasso = Lasso()
print(np.mean(cross_val_score(lr_lasso, X_train, y_train, scoring = "neg_mean_absolute_error", cv=3)))
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

rf_model = RandomForestRegressor(n_estimators=30)

#print(cross_val_score(rf_model, X_train, y_train, scoring="neg_mean_absolute_error", cv=3))

"""
Tuning the Forest
from sklearn.model_selection import GridSearchCV
"""
rf_model.fit(X_train, y_train)

preds = rf_model.predict(X_test)

#print(preds)

from sklearn.metrics import mean_absolute_error

print(mean_absolute_error(y_test, preds))

import pickle
pickl = {'model':rf_model}
pickle.dump(pickl, open{'model_file' + ".p", "wb"})

