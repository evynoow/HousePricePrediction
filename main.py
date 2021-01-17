
# ## Importing Libraries and Data

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

missingvalues = Ames.isnull().sum().sort_values(ascending = False)
missingvalues = missingvalues.reset_index()
missingvalues['Percent'] = missingvalues.iloc[:, 1].apply(lambda x: x*100/len(Ames))
missingvalues.columns = ['Columns', 'MissingValues',"Percent"]
missingvalues = missingvalues[missingvalues['MissingValues'] > 0]
print(missingvalues)

plt.figure(figsize=(13, 6))
sns.set()
sns.barplot(x = 'MissingValues', y = 'Columns', data = missingvalues, palette = 'rocket')


# Handling Missing Data

#deleting spaces and other characters in columns names for selecting easily
Ames.columns = Ames.columns.str.replace(" ","").str.replace("/","")

#defining categorical and numerical values

num_var = ['MSSubClass','LotFrontage','LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd','MasVnrArea',
           'BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea',
           'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd',
           'Fireplaces','GarageYrBlt','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch',
           '3SsnPorch','ScreenPorch','PoolArea','MiscVal','MoSold','YrSold']

cat_var = ['MSZoning','Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood',
           'Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd',
           'MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
           'BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','FireplaceQu',
           'GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','PoolQC', 'Fence', 'MiscFeature',
           'SaleType','SaleCondition']

# create 2 dataframes for numerical and categorical features
df_num = Ames[num_var]
df_cat = Ames[cat_var]
print('df_num', df_num.shape)
print('df_cat', df_cat.shape)


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

