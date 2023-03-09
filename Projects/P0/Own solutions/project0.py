import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd

df_train = pd.read_csv('/home/otps3141/Documents/Dokumente/ETH QE/Semester 2/Intro ML/Projects/P0/task0_sl19d1/train.csv')

### exclude first column
# print(df_train.iloc[:,1:])

y = df_train['y']
# print(y)


y_pred = df_train.iloc[:,2:].mean(axis=1)
# print(df_train.iloc[:,1:].mean(axis=1))

# x = df_train.loc[1:][0:]
# print(x)

### Define train matrix
X = df_train.iloc[:,2:]
print(X)

### Closed form solution
w_star = np.linalg.inv(X.T@X)@(X.T@y)
print(w_star)




RMSE = mean_squared_error(y, y_pred)**0.5
# print(RMSE)