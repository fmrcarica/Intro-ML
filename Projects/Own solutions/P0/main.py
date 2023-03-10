import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd

df_train = pd.read_csv('/home/otps3141/Documents/Dokumente/ETH QE/Semester 2/Intro ML/Projects/P0/task0_sl19d1/train.csv')

### exclude first column
# print(df_train.iloc[:,1:])


### All outcome data
y = df_train['y']
# print(y)


y_pred = df_train.iloc[:,2:].mean(axis=1)
# print(df_train.iloc[:,1:].mean(axis=1))

# x = df_train.loc[1:][0:]
# print(x)

### Define train matrix
X = df_train.iloc[:,2:]
# print(X.shape[1])

### Closed form solution
w_star = np.linalg.inv(X.T@X)@(X.T@y)
# print(w_star)

### Gradient descent approach

w = np.ones((X.shape[1], 1))

# y_array = y.to_numpy()
# array = X.dot(w).to_numpy()

error = y - X.dot(w)
grad = -(X.T).dot(error)[0]
# print(grad)


def grad_desc(X, y, rate = 0.01, iterations = 100):
    w = np.ones((X.shape[1], 1))
    for i in range(iterations):
        errors = y - X.dot(w)[0]
        grad = -1/10000 * (X.T).dot(errors)[0]
        w = w - rate*grad
    return w



# print(grad_desc(X, y))



df_test = pd.read_csv('/home/otps3141/Documents/Dokumente/ETH QE/Semester 2/Intro ML/Projects/P0/task0_sl19d1/test.csv')

X_test = df_test.iloc[:,1:]

print(X_test)

y_outcome = X_test.dot(w_star)


df_sample = pd.read_csv('/home/otps3141/Documents/Dokumente/ETH QE/Semester 2/Intro ML/Projects/P0/task0_sl19d1/sample.csv')

df_sample['y'] = y_outcome

print(df_sample)

df_sample.round(3).to_csv('Projects/P0/task0_sl19d1/sample.csv', index=False)




RMSE = mean_squared_error(y, y_pred)**0.5
# print(RMSE)
