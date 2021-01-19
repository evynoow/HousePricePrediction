
#Importing Libraries and Data

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
import warnings

from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 500)


Ames = pd.read_csv("AmesHousing.csv")


# Exploring Data

print(Ames.shape)
print("Data has 82 columns and 2930 rows.")

print("All columns of dataframe:",Ames.columns)

print(Ames.head(4))

#Dropping order columns which is meaningless for machine learning algorithm.
Ames = Ames.drop(columns = "Order")

print((Ames.apply(lambda a: a.isnull().values.any())).sum(),"columns have missing data")

missingvalues = Ames.isnull().sum().sort_values(ascending = False)  #calculating misisng values
missingvalues = missingvalues.reset_index()
missingvalues['Percent'] = missingvalues.iloc[:, 1].apply(lambda x: x*100/len(Ames)) # calculating The percentage of missing values for each feature
missingvalues.columns = ['Columns', 'MissingValues',"Percent"] #renaming column names
missingvalues = missingvalues[missingvalues['MissingValues'] > 0] #filtering attributes with no missing values
print(missingvalues)

#Plotting missing value count for each attribute
plt.figure(figsize=(13, 6))
sns.set()
sns.barplot(x = 'MissingValues', y = 'Columns', data = missingvalues, palette = 'rocket') #creating plot


# Handling Missing Data (Categorical)

#deleting spaces and other characters in columns names for selecting easily
Ames.columns = Ames.columns.str.replace(" ","").str.replace("/","")

#defining categorical and numerical values

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
num_var = list(Ames.select_dtypes(include=numerics).columns.values) #defining numeric values
cat_var = list(Ames.select_dtypes(include="object").columns.values) #defining categorical values

num_var.remove("PID") #dropping PID value which has no power on prediction
num_var.remove("SalePrice") #dropping SalePrice which will predicted

# create 2 dataframes for numerical and categorical features
df_num = Ames[num_var]
df_cat = Ames[cat_var]

# Missing values of categorical attributes show that the house does not have that property.
# So we can fill in the data as "this feature does not exist"

df_cat.PoolQC=["No Pool" if x is np.nan else x for x in df_cat.PoolQC]
df_cat.FireplaceQu=["No Fireplace" if x is np.nan else x for x in df_cat.FireplaceQu]
df_cat.Alley=["No alley access" if x is np.nan else x for x in df_cat.Alley]
df_cat.Fence=["No Fence" if x is np.nan else x for x in df_cat.Fence]
df_cat.MiscFeature=["None" if x is np.nan else x for x in df_cat.MiscFeature]
df_cat.BsmtQual=["No Basement" if x is np.nan else x for x in df_cat.BsmtQual]
df_cat.BsmtCond=["No Basement" if x is np.nan else x for x in df_cat.BsmtCond]
df_cat.BsmtExposure=["No Basement" if x is np.nan else x for x in df_cat.BsmtExposure]
df_cat.BsmtFinType1=["No Basement" if x is np.nan else x for x in df_cat.BsmtFinType1]
df_cat.BsmtFinType2=["No Basement" if x is np.nan else x for x in df_cat.BsmtFinType2]
df_cat.GarageType=["No Garage" if x is np.nan else x for x in df_cat.GarageType]
df_cat.GarageFinish=["No Garage" if x is np.nan else x for x in df_cat.GarageFinish]
df_cat.GarageCond=["No Garage" if x is np.nan else x for x in df_cat.GarageCond]
df_cat.GarageQual=["No Garage" if x is np.nan else x for x in df_cat.GarageQual]
df_cat.MasVnrType=["None" if x is np.nan else x for x in df_cat.MasVnrType]
df_cat.Electrical=["SBrkr" if x is np.nan else x for x in df_cat.Electrical]

df_cat.isnull().values.any() #Checking if there is nan value left

# Handling Missing Data (Numeric)
missing_num = df_num.isnull().sum().sort_values(ascending = False)
missing_num = missing_num.reset_index()
missing_num['Percent'] = missing_num.iloc[:, 1].apply(lambda x: x*100/len(Ames))
missing_num.columns = ['Columns', 'MissingValues',"Percent"]
missing_num = missing_num[missing_num['MissingValues'] > 0]
print(missing_num)

# Attributes that shows garage and basement areas can be filled with 0.
df_num.BsmtHalfBath = df_num.BsmtHalfBath.fillna(0)
df_num.BsmtFullBath = df_num.BsmtFullBath.fillna(0)
df_num.GarageArea = df_num.GarageArea.fillna(0)
df_num.BsmtFinSF1 = df_num.BsmtFinSF1.fillna(0)
df_num.BsmtFinSF2 = df_num.BsmtFinSF2.fillna(0)
df_num.BsmtUnfSF = df_num.BsmtUnfSF.fillna(0)
df_num.TotalBsmtSF = df_num.TotalBsmtSF.fillna(0)
df_num.GarageCars = df_num.GarageCars.fillna(0)


missing_num = df_num.isnull().sum().sort_values(ascending = False) #calculating missing values
missing_num = missing_num.reset_index()
missing_num['Percent'] = missing_num.iloc[:, 1].apply(lambda x: x*100/len(Ames)) # calculating The percentage of missing values for each feature
missing_num.columns = ['Columns', 'MissingValues',"Percent"] #renaming columns
missing_num = missing_num[missing_num['MissingValues'] > 0] #filtering attributes with no missing values
print(missing_num)

#There is no attribute which have more than %60 missing values.
#We use some methods for filling these values instead of deleting them.
#We can fill these values by calculating mean or median values.
#Distribution plots of each feature will be drawn to decide which one to choose.

df_num_no_null = df_num.dropna()  #We will plot the data so we should filter missing values first.

plt.figure(figsize=(18, 7))
plt.subplot(131) #first plot
sns.distplot(df_num_no_null.LotFrontage) #creating plot

plt.subplot(132) #second plot
sns.distplot(df_num_no_null.MasVnrArea) #creating plot

plt.subplot(133) #third plot
sns.distplot(df_num_no_null.GarageYrBlt) #creating plot
plt.show()

# We can see that LotFrontage attribute is more like normal distribution.
# MasVnrArea and GarageYtBlt is more like skewed distribution.
# Based on these plots, we'll impute the missing values in MasVnrArea and GarageYrBlt by the median.
# The missing values in LotFrontage will be imputed by the mean value.

df_num['LotFrontage'].fillna(df_num['LotFrontage'].mean(), inplace=True) # filling with mean
df_num['MasVnrArea'].fillna(df_num['MasVnrArea'].median(), inplace=True) # filling with median
df_num['GarageYrBlt'].fillna(df_num['GarageYrBlt'].median(), inplace=True) # filling with median

df_num.isnull().values.any() # checking


# Creating dummy variables for categorical attributes
# Creating Dummy Variables
df_cat_dummy = pd.get_dummies(df_cat, drop_first = True) #creates and drops one dummy
df = pd.concat([df_num, df_cat_dummy], axis = 1) # Concatenate dummy cat vars and num vars

print(df.head(2))
print(df.shape)

# Regression Analysis
# To apply regression attributed that will predict should be distribute as normal distribution.

plt.figure(figsize=(14,5)) #figure size
plt.subplot(1,2,1) #first plot
plt.hist(Ames.SalePrice, bins=20, color='b', density=True, label='Sale price') #distribution of sale price
sns.kdeplot(Ames.SalePrice, color='red')
plt.title('Ames Houses Sale Price') #title
plt.xlabel('Price') ; plt.ylabel('#Count') ; plt.legend(loc='upper right')

#Distribution of y is skewed.
# We need to transform it normal distribution. log algorithm is used.

plt.subplot(1,2,2) #Second plot
plt.hist(np.log1p(Ames.SalePrice), bins=20, color='b', density=True, label='Sale price') # distribution of logged sale price
sns.kdeplot(np.log1p(Ames.SalePrice), color='red')
plt.title('Ames Houses Sale Price - Log transformed')
plt.xlabel('log(Price)') ; plt.ylabel('#Count') ; plt.legend(loc='upper right')
plt.show()

X = df
y = np.log1p(Ames['SalePrice'])

# Splitting the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

# Linear Regression Model
regression = LinearRegression()

# Fitting a Linear Regression Model
regression.fit(X_train, y_train)

# Prediction on the test set: Performance Measures
y_pred = regression.predict(X_test)
R2score = r2_score(y_test, y_pred)
print("R^2 : {}".format(R2score))
RMSEscore = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE: {}".format(RMSEscore))


# r^2 decreased and RMSE increased so we get worse results. We will keep outliers.

# Feature Selection
# Correlation plot for all the numeric variables in the dataset
df2 = df_num.copy()
df2['SalePrice'] = Ames.SalePrice
corrmat = df2.corr()
f, ax = plt.subplots(figsize=(15, 12))
_ = sns.heatmap(corrmat, linecolor = 'white', cmap = 'magma', linewidths = 3)

correlations = df_num.corr()
correlations = correlations.iloc[:36, :36]
cut_off = 0.5
high_corrs = correlations[correlations.abs() > cut_off][correlations.abs() != 1].unstack().dropna().to_dict()
high_corrs = pd.Series(high_corrs, index = high_corrs.keys())
high_corrs = high_corrs.reset_index()
high_corrs = pd.DataFrame(high_corrs)
high_corrs.columns = ['Attributes1', 'Attributes2', 'Correlations']
high_corrs['Correlations'] = high_corrs['Correlations'].drop_duplicates(keep = 'first')
high_corrs = high_corrs.dropna().sort_values(by = 'Correlations', ascending = False)
print(high_corrs)
# We can eliminate attributes

# Correlation with the SalePrice variable
corr = df2.corr()['SalePrice']
corr = corr[np.argsort(corr, axis=0)[::-1]]
print(corr)

# Variables highly correlated with SalePrice
k = 10
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df2[cols].values.T)
sns.set(font_scale=1.25)
f, ax = plt.subplots(figsize=(10, 8))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, linewidth = 5,
                 yticklabels=cols.values, xticklabels=cols.values, cmap = 'viridis', linecolor = 'white')
plt.show()

f = pd.melt(Ames, id_vars = 'SalePrice', value_vars = cat_var)
g = sns.FacetGrid(f, col = "variable",  col_wrap = 2, sharex = False, sharey = False, size = 10)
g.map(sns.boxplot, 'value', 'SalePrice', palette = 'viridis')

# Selecting important features based on box plots.
important_cat = ['MSZoning', 'Utilities', 'Neighborhood', 'Condition1', 'Condition2', 'HouseStyle', 'MasVnrType',
           'ExterQual', 'ExterCond', 'Foundation', 'HeatingQC', 'CentralAir', 'KitchenQual',"GarageType" ,'GarageFinish',
           'GarageQual', 'PoolQC', 'SaleType']

# Selecting important numeric features based on correlation with SalePrice.
important_num = list(corrmat.columns)
important_num.remove("SalePrice")

# Dummfiying categorical values
dummfiying = OneHotEncoder(drop='first')

# Merging categorical with numerical values
important_cat_dummfied = pd.DataFrame(dummfiying.fit_transform(df_cat[important_cat]).toarray())
impoartant_df = pd.concat([df_num[important_num],important_cat_dummfied],axis=1)

# Splitting the data into training and testing sets
Xn_train, Xn_test, yn_train, yn_test = train_test_split(impoartant_df, y, test_size = 0.3, random_state = 42)

regression_imp = LinearRegression()
# Fitting a Linear Regression Model
regression_imp.fit(Xn_train, yn_train)

# Evaluating Performance Measures on the test set
y_pred = regression_imp.predict(Xn_test)
R2featured = r2_score(yn_test, y_pred)
print("R^2 : {}".format(R2featured))
RMSEfeatured = np.sqrt(mean_squared_error(yn_test, y_pred))
print("RMSE: {}".format(RMSEfeatured))

# Using only important values increased results so much. We will continue with this dataset.
