

# # Importing all the Dependencies


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


# # original df


# loading the data
df = pd.read_csv('../data/hou_all.csv')
df.head()



#adding columsn name 
col_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV','BIAS_COL']
df.columns = col_names



df = df.iloc[:,:-1]



df.head(2)


# - so there are 505 observations and 15 columns


# lets check columns that are numerical types

numerical_cols = df.select_dtypes(include=[np.number]).columns
numerical_cols


# - looks like no categorical columns
# - also since no categorical value, no encoding required

# # heavily_moderately correlated df


# correlated dataframe
df_correlated = pd.read_csv('../data/highly_moderate_correlated.csv')




df_correlated = df_correlated.iloc[:,1:]



df_correlated.head(2)


# # MEDV outlier removed dataframe
# 


df_removed_outlier= pd.read_csv('../data/df_outlier_removed_from_medv.csv')
df_removed_outlier = df_removed_outlier.iloc[:,1:]
df_removed_outlier.head(2)


# # Some transformations to remove skewness and reduce effect of outliers



from scipy.stats import mstats
# Log Transformation  ==>  to reduce skewness and the impact of high values.
def log_transform(df, columns):
    for col in columns:
        df[col] = np.log(df[col])  # log1p handles log(0) case
    return df

# Square Root Transformation ==> to lessen the influence of high values in a less aggressive manner than log transformation.
def sqrt_transform(df, columns):
    for col in columns:
        df[col] = np.sqrt(df[col])
    return df

# Winsorization ==> to limit extreme values and reduce the effect of outliers.
def winsorize_transform(df, columns):
    for col in columns:
        df[col] = mstats.winsorize(df[col], limits=[0.05, 0.05])  # limits can be adjusted
    return df



# Select columns to apply transformations based on correlation and outlier analysis
log_transform_cols = ['CRIM', 'ZN', 'DIS', 'TAX']
sqrt_transform_cols = ['AGE', 'LSTAT']
winsorize_transform_cols = ['RM', 'NOX', 'PTRATIO', 'RAD', 'B']


# ## transformation for original df


df_transformed = df.copy()



# Apply Log Transformation
df_transformed = log_transform(df_transformed, log_transform_cols)

# Apply Square Root Transformation
df_transformed = sqrt_transform(df_transformed, sqrt_transform_cols)

# Apply Winsorization
df_transformed = winsorize_transform(df_transformed, winsorize_transform_cols)



# df_transformed.to_csv('../data/transformed_original.csv', index=True)


# ## transformation for correlated df


df_transformed_correlated = df_correlated.copy()



# Apply Log Transformation
df_transformed_correlated = log_transform(df_transformed_correlated, log_transform_cols)

# Apply Square Root Transformation
df_transformed_correlated = sqrt_transform(df_transformed_correlated, sqrt_transform_cols)

# Apply Winsorization
df_transformed_correlated = winsorize_transform(df_transformed_correlated, winsorize_transform_cols)



# df_transformed_correlated.to_csv('../data/transformed_correlated.csv', index=True)


# ## transformation for df with no outlier 


df_transformed_removed_outlier = df_removed_outlier.copy()



# Apply Log Transformation
df_transformed_removed_outlier = log_transform(df_transformed_removed_outlier, log_transform_cols)

# Apply Square Root Transformation
df_transformed_removed_outlier = sqrt_transform(df_transformed_removed_outlier, sqrt_transform_cols)

# Apply Winsorization
df_transformed_removed_outlier = winsorize_transform(df_transformed_removed_outlier, winsorize_transform_cols)


# df_transformed_removed_outlier.to_csv('../data/transformed_removed_outlier.csv', index=True)


# # Standarizing the data
# 

# ## standarization for original df

df_scaled = df.copy()



col_to_standarized = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
scaler = StandardScaler()
df_scaled[col_to_standarized] = scaler.fit_transform(df_scaled[col_to_standarized])


# df_scaled.to_csv('../data/scaled_original.csv', index=True)


# ## standarization for correlated df


df_scaled_correlated = df_correlated.copy()


col_to_standarized = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
scaler = StandardScaler()
df_scaled_correlated[col_to_standarized] = scaler.fit_transform(df_scaled_correlated[col_to_standarized])


# df_scaled_correlated.to_csv('../data/scaled_correlated.csv', index=True)


# ## Standarization for no_outlier df

df_scaled_removed_outlier = df_removed_outlier.copy()



col_to_standarized = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
scaler = StandardScaler()
df_scaled_removed_outlier[col_to_standarized] = scaler.fit_transform(df_scaled_removed_outlier[col_to_standarized])


# df_scaled_removed_outlier.to_csv('../data/scaled_no_outlier.csv', index=True)

