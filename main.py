import pandas as pd
import sklearn
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

df = pd.read_csv('Salaries.csv', low_memory=False)
df['Industry'] = df['Industry'].str.strip()
df['Industry'] = pd.Categorical(df['Industry'])
df['Icode'] = df['Industry'].cat.codes
df['Job Title'] = pd.Categorical(df['Job Title'])
df['JTcode'] = df['Job Title'].cat.codes
df['Highest Level of Education Received'] = pd.Categorical(df['Highest Level of Education Received'])
df['EDUcode'] = df['Highest Level of Education Received'].cat.codes
df['City'] = pd.Categorical(df['City'])
df['CITYcode'] = df['City'].cat.codes

target_column = ['Salary']
predictors = list(df[['Icode', 'JTcode', 'EDUcode', 'CITYcode']])
df[predictors] = df[predictors]/df[predictors].max()

X = df[predictors].values
y = df[target_column].values

array = df.to_numpy()

print(array)

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=0)


regressor = LinearRegression()
regressor.fit(xtrain, ytrain)

y_pred = regressor.predict(xtest)

print("Predicted Results: ")
print(y_pred)
print(np.sqrt(mean_squared_error(ytest, y_pred)))
print(r2_score(ytest, y_pred))

rr = Ridge(alpha=0.01)
rr.fit(xtrain, ytrain)
pred_train_rr = rr.predict(xtrain)
print(np.sqrt(mean_squared_error(ytrain, pred_train_rr)))
print(r2_score(ytrain, pred_train_rr))

pred_test_rr = rr.predict(xtest)
print(np.sqrt(mean_squared_error(ytest, pred_test_rr)))
print(r2_score(ytest, pred_test_rr))