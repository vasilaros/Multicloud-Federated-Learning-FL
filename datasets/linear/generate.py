# Linear regression dataset
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
import pandas as pd

def f(x, a, b, noise=True):
    n = np.random.normal(0,1e-6)
    return a*x+b + n


# Generate data

X = []
Y = []

for i in range(70000):
    X.append(i)
    Y.append(f(i, 1.0/100000, 1.0/80000))



data = np.array([X,Y])

X = np.array(X).reshape(-1, 1)
Y = np.array(Y).reshape(-1, 1)
lr = LinearRegression()  # create object for the class
lr.fit(X, Y)  # perform linear regression

#plt.plot(X,Y)

print(lr.coef_[0])
print(lr.intercept_[0])



df = pd.DataFrame(np.column_stack((X,Y)))
df.columns = ['X', 'Y']
plt.plot(df.X, df.Y)

df.to_excel('data.xlsx',index=False)