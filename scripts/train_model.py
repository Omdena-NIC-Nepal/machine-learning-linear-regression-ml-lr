#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Beginner Project: Linear Regression



# imprting all the packages
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns



# loading dataset
df = pd.read_csv('../data/hou_all.csv', header=None)
df.head()


#adding column name 
col_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV','BIAS_COL']
df.columns = col_names


# - excluding the target variable 



df.head()



df = df.iloc[:,:-1]



df.head()



# defining feature and target
# feature
X = df.drop(columns=['MEDV'], axis=1)
# target
y = df['MEDV']



X.head(2)


y.head(2)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)



model = LinearRegression()
model.fit(X_train, y_train)




X_test



y_test



# predicting training set
pred = model.predict(X_test)

# performing cross validation
mae = mean_absolute_error(y_test, pred)
mse = mean_squared_error(y_test, pred)
r2 = r2_score(y_test, pred)


print("Linear Regression -\n  MAE:", mae, "MSE:" ,mse, "R2:" ,r2)


# - Mean Squared Error (MSE): It measures the average squared difference between the actual and predicted values. Lower MSE indicates better model performance.
# The ideal value of both MSE  is 0. The model gives the best result when the value are lower. Here,  MSE has significantly larger value i.e 19.573881758547856. MSE penalizes larger errors more than smaller ones due to squaring the differences. 
# 
# - Mean Absolute Error (MAE): It measures the average absolute difference between the actual and predicted values.
# Lower MAE indicates better model performance.
# The ideal value of both MAE is 0. The model gives the best result when the value are lower. Here, while the value of MAE is l3.182394167629652 .This means, on average, the model's predictions are off by about 3.18 units from the actual values. 
# 
# - R-squared (R2): It represents the proportion of the variance in the dependent variable that is predictable from the independent variables.
# R2 ranges from 0 to 1, with higher values indicating better model performance.
# 
# The ideal value of both R-squares is 1. The model gives the best result when the value as high as possible, closer to 1. Here, the value of R2 is 6667024777797158. This means that about 67% of the variance in the target variable is explained by the model.



plt.figure(figsize=(10, 6))
plt.scatter(y_test, pred, color='blue', edgecolors='k', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Linear Regression: Actual vs Predicted')
plt.show()


# 
# - Points that are close to the diagonal line indicate good predictions by the model. The closer the points are to the line, the better the model's performance.
# - Points that are far from the diagonal line indicate larger errors in the predictions.The farther the points are from the line, the worse the model's performance for those observations.

# ## Appying Cross validation


# Perform cross-validation
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(model, X_train, y_train, cv=6)



# getting best score
print(f"Mean cross-validation score: {np.mean(cv_scores)}")


# The mean cross-validation score is approximately 0.728. This score represents the average performance of your model across the 6 folds of the cross-validation process

# #### Hyperparameter 
# For this, using Grid-search. It is used to find the optimal hyperparameters of a model which results in the most ‘accurate’ predictions.

# ## Using Ridge


# Defining parameter grid for Ridge regression
param_grid = {'alpha': [0.1, 1, 10, 100, 200]}
# Initialize the Ridge regression model
ridge = Ridge()




# Perform Grid Search with cross-validation
grid_search = GridSearchCV(ridge, param_grid, cv=5)
grid_search.fit(X_train, y_train)



# Get the best model from the grid search
best_ridge_model = grid_search.best_estimator_
# Make predictions on the test set with the best model
pred_best = best_ridge_model.predict(X_test)



# Evaluate the best model's performance
mae_best = mean_absolute_error(y_test, pred_best)
mse_best = mean_squared_error(y_test, pred_best)
r2_best = r2_score(y_test, pred_best)

print(f"Ridge Regression - Best Alpha: {grid_search.best_params_['alpha']}, MAE: {mae_best}, MSE: {mse_best}, R2: {r2_best}")


# - These results are very close to those of the plain linear regression model, suggesting that the regularization effect of Ridge regression was minimal in this case.

# ## Using Lasso



param_lasso = {'alpha': [0.01, 0.1, 1, 10, 100]}

# Create a Lasso model
lasso = Lasso()

# Create GridSearchCV object
grid_search_lasso = GridSearchCV(estimator=lasso, param_grid=param_lasso, cv=5, scoring='r2')

# Perform grid search on the training data
grid_search_lasso.fit(X_train, y_train)

# Print the best parameters and the best score
print(f"Best parameters: {grid_search_lasso.best_params_}")
print(f"Best cross-validation score: {grid_search_lasso.best_score_}")

# Evaluate the best model on the test data
best_lasso = grid_search_lasso.best_estimator_
y_pred = best_lasso.predict(X_test)

print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))






