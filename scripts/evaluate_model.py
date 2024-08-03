from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#getting list of dataset
import os

data_dir = 'data'  
csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

csv_files

def perform_linear_regression(file_path):
    # Load dataset
    df = pd.read_csv(file_path, header=None)
    
    # Add column names
    col_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV', 'BIAS_COL']
    df.columns = col_names

     # Remove non-numeric columns and handle potential issues with headers
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    
    # Remove the last column
    df = df.iloc[:, :-1]

    # Replace infinite values with NaN and then drop them
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    # Split the data into features and target
    X = df.drop('MEDV', axis=1)
    y = df['MEDV']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    cv_score = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 10).mean()

    
    # Store results in a dictionary
    results = {
        'file': os.path.basename(file_path),
        'MAE': mae,
        'MSE': mse,
        'R2': r2,
        'cross_val_score': cv_score
    }
    
    return results


# Process all CSV files and store results
results_list = []

for csv_file in csv_files:
    file_path = os.path.join(data_dir, csv_file)
    results = perform_linear_regression(file_path)
    results_list.append(results)

# Create DataFrame from results
results_df = pd.DataFrame(results_list)




results_df

import matplotlib.pyplot as plt
import numpy as np

# Define colors for different metrics
colors_mae = 'skyblue'
colors_mse = 'lightgreen'
colors_r2 = 'salmon'
colors_cv = 'mediumpurple'

# Set up the matplotlib figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Metrics Comparison Across Datasets', fontsize=16)

# Plot the MAE scores
axes[0, 0].bar(results_df['file'], results_df['MAE'], color=colors_mae)
axes[0, 0].set_title('Mean Absolute Error')
axes[0, 0].set_xlabel('Dataset')
axes[0, 0].set_ylabel('MAE')
axes[0, 0].tick_params(axis='x', rotation=90)

# Plot the MSE scores
axes[0, 1].bar(results_df['file'], results_df['MSE'], color=colors_mse)
axes[0, 1].set_title('Mean Squared Error')
axes[0, 1].set_xlabel('Dataset')
axes[0, 1].set_ylabel('MSE')
axes[0, 1].tick_params(axis='x', rotation=90)

# Plot the R2 scores
axes[1, 0].bar(results_df['file'], results_df['R2'], color=colors_r2)
axes[1, 0].set_title('R2 Score')
axes[1, 0].set_xlabel('Dataset')
axes[1, 0].set_ylabel('R2')
axes[1, 0].tick_params(axis='x', rotation=90)

# Ensure 'cross_val_score' column exists in the DataFrame
if 'cross_val_score' in results_df.columns:
    # Plot the Cross Validation scores
    axes[1, 1].bar(results_df['file'], results_df['cross_val_score'], color=colors_cv)
    axes[1, 1].set_title('Cross Validation Score')
    axes[1, 1].set_xlabel('Dataset')
    axes[1, 1].set_ylabel('Cross Validation Score')
    axes[1, 1].tick_params(axis='x', rotation=90)

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


sorted_df = results_df.sort_values(by=['MAE', 'MSE', 'R2','cross_val_score'], ascending=[True, True, False,False])

sorted_df

# Load dataset
df = pd.read_csv(file_path, header=None)
    
    # Add column names
col_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV', 'BIAS_COL']
df.columns = col_names

     # Remove non-numeric columns and handle potential issues with headers
df = df.apply(pd.to_numeric, errors='coerce').dropna()
    
    # Remove the last column
df = df.iloc[:, :-1]

    # Replace infinite values with NaN and then drop them
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
    
    # Split the data into features and target
X = df.drop('MEDV', axis=1)
y = df['MEDV']
    
    # Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#====================================================NORMAL WAY ===========================================================================

mae = []
mse = []
r2 = []
# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
r2.append(r2_score(y_test, lr_pred))
mae.append(mean_absolute_error(y_test, lr_pred))
mse.append(mean_squared_error(y_test, lr_pred))

# Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
r2.append(r2_score(y_test, ridge_pred))
mae.append(mean_absolute_error(y_test, ridge_pred))
mse.append(mean_squared_error(y_test, ridge_pred))

# Lasso Regression
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)
r2.append(r2_score(y_test, lasso_pred))
mae.append(mean_absolute_error(y_test, ridge_pred))
mse.append(mean_squared_error(y_test, ridge_pred))

df_metrics = pd.DataFrame({'MAE':mae, 'MSE':mse,'R2':r2}, index=['linear','Ridge','Lasso']) 


# =======================================================GRID SEARCH CV=====================================================
# Define parameter grids
param_grid_lr = {}
param_grid_ridge = {'alpha': [0.1, 1, 10, 100]}
param_grid_lasso = {'alpha': [0.1, 1, 10, 100]}

# Initialize models
lr = LinearRegression()
ridge = Ridge()
lasso = Lasso()

# Grid Search CV for Linear Regression (no hyperparameters to tune)
grid_search_lr = GridSearchCV(estimator=lr, param_grid=param_grid_lr, cv=5, scoring='r2')
grid_search_lr.fit(X_train, y_train)

# Grid Search CV for Ridge Regression
grid_search_ridge = GridSearchCV(estimator=ridge, param_grid=param_grid_ridge, cv=5, scoring='r2')
grid_search_ridge.fit(X_train, y_train)

# Grid Search CV for Lasso Regression
grid_search_lasso = GridSearchCV(estimator=lasso, param_grid=param_grid_lasso, cv=5, scoring='r2')
grid_search_lasso.fit(X_train, y_train)

# Get the best models
best_lr = grid_search_lr.best_estimator_
best_ridge = grid_search_ridge.best_estimator_
best_lasso = grid_search_lasso.best_estimator_

# Initialize lists to store metrics
mae_grid = []
mse_grid = []
r2_grid = []

# Linear Regression
lr_pred_grid = best_lr.predict(X_test)
r2_grid.append(r2_score(y_test, lr_pred_grid))
mae_grid.append(mean_absolute_error(y_test, lr_pred_grid))
mse_grid.append(mean_squared_error(y_test, lr_pred_grid))

# Ridge Regression
ridge_pred_grid= best_ridge.predict(X_test)
r2_grid.append(r2_score(y_test, ridge_pred_grid))
mae_grid.append(mean_absolute_error(y_test, ridge_pred_grid))
mse_grid.append(mean_squared_error(y_test, ridge_pred_grid))

# Lasso Regression
lasso_pred_grid= best_lasso.predict(X_test)
r2_grid.append(r2_score(y_test, lasso_pred_grid))
mae_grid.append(mean_absolute_error(y_test, lasso_pred_grid))
mse_grid.append(mean_squared_error(y_test, lasso_pred_grid))

# Create DataFrame for metrics
df_metrics_grid= pd.DataFrame({'MAE':mae_grid, 'MSE':mse_grid, 'R2':r2_grid}, index=['Linear', 'Ridge', 'Lasso'])






df_metrics


# Set up the matplotlib figure with subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Model Metrics Comparison', fontsize=16)

# Plot the MAE scores
axes[0].bar(df_metrics.index, df_metrics['MAE'], color='skyblue')
axes[0].set_title('Mean Absolute Error (MAE)')
axes[0].set_xlabel('Model')
axes[0].set_ylabel('MAE')
axes[0].tick_params(axis='x', rotation=45)

# Plot the MSE scores
axes[1].bar(df_metrics.index, df_metrics['MSE'], color='lightgreen')
axes[1].set_title('Mean Squared Error (MSE)')
axes[1].set_xlabel('Model')
axes[1].set_ylabel('MSE')
axes[1].tick_params(axis='x', rotation=45)

# Plot the R2 scores
axes[2].bar(df_metrics.index, df_metrics['R2'], color='salmon')
axes[2].set_title('R2 Score')
axes[2].set_xlabel('Model')
axes[2].set_ylabel('R2')
axes[2].tick_params(axis='x', rotation=45)

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# Create subplots for Actual vs Predicted in the same figure
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Actual vs Predicted Comparison', fontsize=16)

# Linear Regression
axes[0].scatter(y_test, lr_pred, color='blue', edgecolors='k', alpha=0.7, label='Predicted')
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3, label='Ideal')
axes[0].set_xlabel('Actual')
axes[0].set_ylabel('Predicted')
axes[0].set_title('Linear Regression: Actual vs Predicted')
axes[0].legend()

# Ridge Regression
axes[1].scatter(y_test, ridge_pred, color='green', edgecolors='k', alpha=0.7, label='Predicted')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3, label='Ideal')
axes[1].set_xlabel('Actual')
axes[1].set_ylabel('Predicted')
axes[1].set_title('Ridge Regression: Actual vs Predicted')
axes[1].legend()

# Lasso Regression
axes[2].scatter(y_test, lasso_pred, color='red', edgecolors='k', alpha=0.7, label='Predicted')
axes[2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3, label='Ideal')
axes[2].set_xlabel('Actual')
axes[2].set_ylabel('Predicted')
axes[2].set_title('Lasso Regression: Actual vs Predicted')
axes[2].legend()

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


df_metrics_grid


# Set up the matplotlib figure with subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Model Metrics Comparison', fontsize=16)

# Plot the MAE scores
axes[0].bar(df_metrics_grid.index, df_metrics_grid['MAE'], color='skyblue')
axes[0].set_title('Mean Absolute Error (MAE)')
axes[0].set_xlabel('Model')
axes[0].set_ylabel('MAE')
axes[0].tick_params(axis='x', rotation=45)

# Plot the MSE scores
axes[1].bar(df_metrics_grid.index, df_metrics_grid['MSE'], color='lightgreen')
axes[1].set_title('Mean Squared Error (MSE)')
axes[1].set_xlabel('Model')
axes[1].set_ylabel('MSE')
axes[1].tick_params(axis='x', rotation=45)

# Plot the R2 scores
axes[2].bar(df_metrics_grid.index, df_metrics_grid['R2'], color='salmon')
axes[2].set_title('R2 Score')
axes[2].set_xlabel('Model')
axes[2].set_ylabel('R2')
axes[2].tick_params(axis='x', rotation=45)

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# Create subplots for Actual vs Predicted in the same figure
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Actual vs Predicted Comparison', fontsize=16)

# Linear Regression
axes[0].scatter(y_test, lr_pred_grid, color='blue', edgecolors='k', alpha=0.7, label='Predicted')
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3, label='Ideal')
axes[0].set_xlabel('Actual')
axes[0].set_ylabel('Predicted')
axes[0].set_title('Linear Regression: Actual vs Predicted')
axes[0].legend()

# Ridge Regression
axes[1].scatter(y_test, ridge_pred_grid, color='green', edgecolors='k', alpha=0.7, label='Predicted')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3, label='Ideal')
axes[1].set_xlabel('Actual')
axes[1].set_ylabel('Predicted')
axes[1].set_title('Ridge Regression: Actual vs Predicted')
axes[1].legend()

# Lasso Regression
axes[2].scatter(y_test, lasso_pred_grid, color='red', edgecolors='k', alpha=0.7, label='Predicted')
axes[2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3, label='Ideal')
axes[2].set_xlabel('Actual')
axes[2].set_ylabel('Predicted')
axes[2].set_title('Lasso Regression: Actual vs Predicted')
axes[2].legend()

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


df_metrics

df_metrics_grid



