import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(12345)

data = pd.read_csv("assignment1_data.csv") 
#print(data.head())     
data=data.to_numpy()
x=data[:,0];
y=data[:,1];
Rx = np.array([[5, -1, -2], [ -1, 5, -1],[ -2, -1, 5]]);
ryx = np.array([[1],[5.3],[-3.9]]);
alpha = 0.001;
N = np.size(x)
length = 3;
w = np.zeros((length, 1))
x_delayed = np.zeros((length, 1))
w_history = np.zeros((1,length))
e_history = []
J_history =[]
w0 = np.array([[0.2],[1],[-0.5]])

for i in range(N):
    x_delayed = np.array([[x[i]],[0],[0]])
    y_hat = np.transpose(w).dot(x_delayed)
    e = y[i] - y_hat
    e_history.append(e)
    w_history=np.append(w_history, np.transpose(w), axis=0)
    w = w - alpha * (-2*(ryx - Rx.dot(w)))
    j_his = y[i]*y[i] - np.transpose(ryx).dot(np.linalg.inv(Rx)).dot(ryx)+np.transpose(w-w0).dot(Rx).dot(w-w0)
    J_history = np.append(J_history, j_his) #GD
    #w = w + 2 * alpha * np.linalg.inv(Rx).dot(ryx - Rx.dot(w)) #Newton
    

w0 = w_history[:,0]
w3 = w_history[:,2]
# plt.plot(w0, w3)
# plt.show()


# X1, X2 = np.meshgrid(x1, x2)
# Y = np.sqrt(np.square(X1) + np.square(X2))


# levels = [0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 14.0]
# cp = plt.contour(X1, X2, Y, levels, colors='black', linestyles='dashed', linewidths=1)
# plt.clabel(cp, inline=1, fontsize=10)
# cp = plt.contourf(X1, X2, Y, levels)
# plt.xlabel('X1')
# plt.ylabel('X2')

# print('finish')


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        