import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import KBinsDiscretizer

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import BayesianRidge

data_file = pd.read_csv('./vgsales.csv')
# print(data_file.head(3))

y = data_file.pop('Global_Sales').values

platform_counts = data_file['Platform'].value_counts()
# print("Platform Counts:")
# print(platform_counts)

si_step = ('si', SimpleImputer(strategy='constant', fill_value='MISSING'))
ohe_step = ('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore'))
steps = [si_step, ohe_step]
pipe = Pipeline(steps)

string_cols = ['Platform', 'Genre', 'Publisher']
string_train = data_file[string_cols]
# print(string_train.head(3))

string_train_transformed = pipe.fit_transform(string_train)
# print("Shape:", string_train_transformed.shape)

year_si_step = ('si', SimpleImputer(strategy='median'))
kbd_step = ('kbd', KBinsDiscretizer(encode='onehot-dense'))
steps = [year_si_step, kbd_step]
year_pipe = Pipeline(steps)
year_cols = ['Year']
year_train = data_file[year_cols]
print(year_train.head(3))

year_train_transformed = year_pipe.fit_transform(year_train)
print("Num per bin:", year_train_transformed.sum(axis=0))
