 # Importing all the Dependencies



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


# # Loading the data and Preprocessing

# ## About the data set
# This is a Data Set from UCI Machine Learning Repository which concerns housing values in suburbs of Boston.
# - Link = https://www.kaggle.com/datasets/heptapod/uci-ml-datasets/data
# 
# 
# Features (columns):
# - CRIM: per capita crime rate by town.
# - ZN: proportion of residential land zoned for lots over 25,000 sq. ft.
# - INDUS: proportion of non-retail business acres per town.
# - CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise).
# - NOX: nitrogen oxides concentration (parts per 10 million).
# - RM: average number of rooms per dwelling.
# - AGE: proportion of owner-occupied units built prior to 1940.
# - DIS: weighted distances to five Boston employment centers.
# - RAD: index of accessibility to radial highways.
# - TAX: full-value property tax rate per $10,000.
# - PTRATIO: pupil-teacher ratio by town.
# - B: 1000(Bk - 0.63)^2 where Bk is the proportion of black residents by town.
# - LSTAT: percentage of lower status of the population.
# - MEDV: median value of owner-occupied homes in $1000s (Target).
# 



# loading the data
df = pd.read_csv('../data/hou_all.csv')
df.head()




#adding columsn name 
col_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV','BIAS_COL']
df.columns = col_names




df = df.iloc[:,:-1] #removing BIAS column




df.head(2)




df.columns



df.shape


# - so there are 505 observations and 15 columns



df.info()



# lets check null values
print("The number of null values across each column")
df.isna().sum()


# - There are no null values. So there is no need to drop anything 



# lets check columns that are numerical types

numerical_cols = df.select_dtypes(include=[np.number]).columns
numerical_cols



# categorical columns
categorical_cols = df.select_dtypes(include=['object','category']).columns
categorical_cols


# - looks like no categorical columns
# - also since no categorical value, no encoding required



# checking summary statistics
df.describe()


# - we will create a different dataset without outliers



#checking the distribution of target variable
sns.distplot(df["MEDV"], bins = 25, axlabel = "House Price")
plt.title("MEDV Distribution")
plt.show()


# - The distribution of the MEDV (Median House Price) is slightly right-skewed, with the majority of house prices concentrated between 15 and 30. There are a few extreme values above 50, indicating the presence of high-value outliers.

# # Univariant Analysis
# 



# Histograms for numerical features
df.hist(bins=20, figsize=(20, 15))
plt.show()



# Interpretation of Each Histogram:
# - CRIM: The distribution is highly right-skewed, with most values clustered close to zero.
# - ZN: Most values are concentrated at zero, indicating many areas have no large residential lots.
# - INDUS: The distribution looks like bimodal, with peaks around 7 and 18. Indicates two distinct groups in terms of non-retail business acres per town.
# - CHAS: This looks like binary variable with high concentration at 0 and low at 1
# - NOX: The distribution is somewhat uniform with a peak around 0.5.
# - RM: The distribution is fairly normal with a peak around 6 to 7 rooms per dwelling. Indicates that most homes have between 5 to 7 rooms.
# - AGE: The distribution is right-skewed with a peak at 100. Many houses are older.
# - DIS: The distribution is right-skewed, indicating most towns have shorter distances to employment centers.
# - RAD: The distribution shows a peak at 24.
# - TAX: The distribution is bimodal, with peaks around 300 and 700. Indicates distinct groups in terms of property tax rates.
# - PTRATIO: The distribution is left-skewed, with most values concentrated around 20.
# - B: The distribution is highly right-skewed, with a peak close to 400. Indicates a high proportion of towns with nearly all black residents (high B value).
# - LSTAT: The distribution is right-skewed with a peak around 5. Indicates most towns have a lower percentage of lower status population.
# - MEDV: The distribution is right-skewed, with most house prices ranging from 15 to 30.
#         



df['CHAS'].value_counts()  


# - looks like CHAS can act as categorical



# Value Counts (only if you have categorical features, replace 'CHAS' with actual categorical columns)
plt.figure(figsize=(10, 6))
sns.barplot(df['CHAS'].value_counts())
plt.title('Count Plot of CHAS')
plt.show()


# - looks like the column CHAS has just to unique values with 0 being highest proportion and 1 being lowest proportion.


print("Looking closely to distributions with KDE plot")
plt.figure(figsize=(20, 15))
for i, col in enumerate(df.columns):
    plt.subplot(5, 3, i + 1)
    sns.kdeplot(df[col])
    plt.title(f'KDE Plot of {col}')
plt.tight_layout()
plt.show()


# ## Box plot and outliers detection



# Boxplots for numerical features
plt.figure(figsize=(20, 15))
for i, col in enumerate(df.columns):
    plt.subplot(5, 3, i + 1)
    sns.boxplot(y=df[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()


# - looks like columns like CRIM, ZN, CHAS, RM, DIS, PTRATIO, B, LSTAT, and MEDV have outliers
# - we will delete outliers in preprocessing stage

# ## Outlier detection using IQR method



def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

for col in numerical_cols:
    outlier = detect_outliers_iqr(df, col)

    print(f'The shape of outlier for column {col} is {outlier.shape}')
    print(f'The percentage of outlier for column {col} is {(outlier.shape[0]/df.shape[0]) * 100}')
    print("=============================================================")
    


# ### for now just remove the outlier for target value
# 



df_outlier_removed_from_medv= remove_outliers_iqr(df,'MEDV')



df_outlier_removed_from_medv.to_csv('../data/df_outlier_removed_from_medv.csv', index=True)


# - other outlier, we will see in preporcessing step

# # Bivariant analysis


# Scatter plots for numerical features against the target (MEDV)
print("Scatter plots for numerical features against the target")
plt.figure(figsize=(20, 15))
for i, col in enumerate(df.columns):
    if col != 'MEDV':
        plt.subplot(5, 3, i + 1)
        sns.scatterplot(x=df[col], y=df['MEDV'])
        plt.title(f'Scatter plot of {col} vs MEDV')
plt.tight_layout()
plt.show()


# Interpretation of Each Scatter Plot:
# - CRIM vs MEDV: Higher crime rates are associated with lower house prices.
# - ZN vs MEDV: There is no clear relationship between the proportion of residential land zoned.
# - INDUS vs MEDV: Higher proportion of non-retail business acre per town tend to have lower house prices.Some areas with low proportion also have low house prices, showing varied distribution.
# - CHAS vs MEDV: Houses bordering the Charles River (CHAS = 1) have a wider range of prices, often higher. Most data points are for areas not bordering the river (CHAS = 0).
# - NOX vs MEDV: Higher nitrogen oxide concentrations are associated with lower house prices.
# - RM vs MEDV: There is a strong positive correlation between the number of rooms and house prices. Houses with more rooms tend to have higher prices.
# -AGE vs MEDV: There is no clear relationship between the age of the houses and their prices. House prices remain relatively constant across different age values.
# - DIS vs MEDV: Greater distances to employment centers are associated with lower house prices. House prices tend to vary for shorter distances.
# - RAD vs MEDV: No clear relationship between accessibility to radial highways and house prices.
# - TAX vs MEDV: Higher property tax rates are associated with lower house prices. There are clusters of data points with different tax rates showing varied house prices.
# - PTRATIO vs MEDV: Higher pupil-teacher ratios are generally associated with lower house prices.
# - B vs MEDV: Higher values of 'B' (proportion of black residents) are associated with higher house prices. There are few high house prices at lower 'B' values, indicating some variation.
# - LSTAT vs MEDV: There is a strong negative correlation between the percentage of lower status population and house prices. Higher LSTAT values are associated with lower house prices.
# 
# 
# 



# Joint Plots
joint_plot_col = ['CRIM','RM','TAX','LSTAT']
for col in joint_plot_col:
    if col != 'MEDV':
        sns.jointplot(x=df[col], y=df['MEDV'], kind='reg')
        plt.suptitle(f'Joint plot of {col} vs MEDV', y=1.02)
        plt.show()


# Interpretation of Joint Plot: CRIM vs MEDV 
# 
# - Negative Correlation: There is a clear negative correlation between the crime rate (CRIM) and the median house prices (MEDV), as indicated by the downward slope of the regression line.
# - Concentration and Outliers: Most data points are concentrated near lower crime rates (CRIM < 20) and higher house prices (MEDV > 20). A few outliers exist with high crime rates and low house prices, confirming the skewness observed in the individual distributions
# 
# 
# Interpretation of Joint Plot: RM vs MEDV
# 
# - Positive Correlation: There is a strong positive correlation between the average number of rooms per dwelling (RM) and the median house prices (MEDV), as indicated by the upward slope of the regression line. Houses with more rooms tend to have higher prices.
# - Distribution: The scatter plot shows that most data points cluster around RM values between 5 and 7, with corresponding house prices ranging from $10,000 to $30,000. The histograms on the top and right show the distribution of RM and MEDV, respectively, confirming the normal distribution for RM and a right-skewed distribution for MEDV.
# 
# Interpretation of Joint Plot: TAX vs MEDV
# 
# - Negative Correlation: There is a negative correlation between property tax rate (TAX) and median house prices (MEDV), as indicated by the downward slope of the regression line. Higher tax rates are generally associated with lower house prices.
# - Distribution: The scatter plot shows a concentration of data points around lower tax values (200-400) with varied house prices. The histograms on the top and right show the distribution of TAX and MEDV, respectively, with TAX values clustering around specific points (200, 400, 700) and MEDV showing a right-skewed distribution.
# 
# Interpretation of Joint Plot: LSTAT vs MEDV
# 
# - Strong Negative Correlation: There is a strong negative correlation between the percentage of lower status population (LSTAT) and median house prices (MEDV), as indicated by the downward slope of the regression line. Higher LSTAT values are associated with lower house prices.
# -Distribution: The scatter plot shows that as LSTAT increases, MEDV generally decreases. The histograms on the top and right show the distribution of LSTAT and MEDV, respectively, with LSTAT values somewhat right-skewed and MEDV showing a right-skewed distribution.

# # Multivariant analysis


# check check co-relation between columns
corr = df.corr()
corr


# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


print("Printing correlation across target variable: MEDV")

corr['MEDV'].sort_values(ascending=False)


# Interpretation of Correlation with MEDV
# 
# The table shows the Pearson correlation coefficients of various features with the target variable, MEDV (Median House Value), sorted in descending order of their correlation values.
# - Strong Positive Correlation: RM (0.695): The number of rooms per dwelling shows a strong positive correlation with house prices. More rooms generally indicate higher house prices.
# - Moderate Positive Correlations: ZN (0.360): The proportion of residential land zoned for large lots has a moderate positive correlation with house prices.
# B (0.334): The proportion of black residents has a moderate positive correlation with house prices, suggesting that higher values are associated with higher house prices.
# - Moderate Negative Correlations: NOX (-0.427): Nitrogen oxide concentration has a moderate negative correlation with house prices, suggesting that higher pollution levels are associated with lower house prices. TAX (-0.469): Property tax rate has a moderate negative correlation with house prices. INDUS (-0.484): Proportion of non-retail business acres per town has a moderate negative correlation with house prices. PTRATIO (-0.508): Pupil-teacher ratio has a moderate negative correlation with house prices.
# - Strong Negative Correlation: LSTAT (-0.738): The percentage of the lower status population has a strong negative correlation with house prices. Higher LSTAT values are strongly associated with lower house prices.



corr_df = ['RM','ZN','B','NOX','TAX','INDUS','LSTAT','MEDV']
df_correlated = df[corr_df]

df_correlated.head()


df.to_csv('../data/highly_moderate_correlated.csv',index=True)


# Pair plot for multivariate analysis
print("Pairplot for multivariate analysis")
sns.pairplot(df)
plt.show()


# Assuming 'CHAS' is a categorical feature, we will use it for Facet Grid
g = sns.FacetGrid(df, col="CHAS", margin_titles=True, height=5)
g.map(sns.histplot, "MEDV", kde=True)
plt.show()