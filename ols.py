import numpy as np
import matplotlib.pyplot as plt

def hit(x,y,w, thresh=0.05):
    y_hat = w[0]*x+w[1]
    hit = np.sum((np.abs((y-y_hat)/y_hat)<thresh).astype(int))/len(x)
    return hit

x = np.arange(30)
y = np.zeros(len(x))
for i in x:
    y[i] = 1e-6*x[i] + 1.25e-6 + .5e-6*np.random.rand()


x0 = np.column_stack((x, np.ones(len(x))))
w = np.linalg.inv(x0.T @ x0) @ x0.T @ y

plt.figure()
plt.plot(x,y,'r')
plt.scatter(x, w[0]*x+w[1])

print(f'Hit ratio = {hit(x,y,w):.2g}')

mean_loss = np.mean(y - w[0]* - w[1])
print(f'Mean loss = {mean_loss:.2g}')