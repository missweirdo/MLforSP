import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

### Data ###
data = pd.read_csv("assignment1_data.csv")  
data=data.to_numpy()
x=data[:,0];
y=data[:,1];

### Input ###
Rx = np.array([[5, -1, -2], [ -1, 5, -1],[ -2, -1, 5]]);
ryx = np.array([[1],[5.3],[-3.9]]);
alpha = 0.001;
length = 3;

### Var ###
N = np.size(x)
Rx_inv = np.linalg.inv(Rx)
w0 = np.dot(Rx_inv, ryx)
w = np.zeros((length, 1))
x_delayed = np.zeros((length, 1))
w_history = np.zeros((1,length))
e_history = []

gamma = 1 - (10**-4)
epsilon = 0.000001;

for i in range(N):
    x_delayed = np.vstack([x[i], x_delayed[0:length-1]])
    y_hat = np.transpose(w).dot(x_delayed)
    e = y[i] - y_hat
    e_history.append(e)
    w_history=np.append(w_history, np.transpose(w), axis=0)
    sigma = (np.dot(np.transpose(x_delayed),x_delayed)/length)+epsilon;
    ### Different Methods ###
    
    #w = w - alpha * (-2*(ryx - Rx.dot(w))) #SGD
    #w = w + 2 * alpha * Rx_inv.dot(ryx - Rx.dot(w)) #Newton
    #w = w + 2 * alpha * np.dot(x_delayed,e) #LMS
    #w = w + ((2 * alpha)/sigma) * np.dot(x_delayed,e);#NLMS
    Rx_inv_old = Rx_inv;
    ryx_old = ryx;
    g = ((Rx_inv_old * x_delayed) / (gamma**2 + np.transpose(x_delayed) * Rx_inv_old * x_delayed));
    Rx_inv = gamma**-2 * (Rx_inv_old - np.dot(g,np.transpose(x_delayed)) * Rx_inv_old);
    ryx = gamma**2 * ryx_old + np.dot(x_delayed * y[i]);
    w = np.dot(Rx_inv, ryx);#RLS


### contourplot ###
w_0 = w_history[1:,0]
w_1 = w_history[1:,1]
w_2 = w_history[1:,2]

wiener0 = w0[0]
wiener1 = w0[1]

n_size = 100
scale_factor = 1
levels = 13

xlist = np.linspace(wiener0 - scale_factor, wiener0 + scale_factor, n_size)
ylist = np.linspace(wiener1 - scale_factor, wiener1 + scale_factor, n_size)
X, Y = np.meshgrid(xlist, ylist)

### calculating J ###
J = np.zeros((n_size,n_size))
E_y_sq = np.mean(np.square(y))
J_min = E_y_sq-np.dot(np.dot(np.transpose(ryx),Rx_inv),ryx)

for i in range(n_size):
    for j in range(n_size):
        xx=X[i,j]
        yy=Y[i,j]
        w_contour = np.array([[xx], [yy], [-0.5]])
        ww0 = w_contour-w0
        J[i, j] =J_min + np.dot(np.dot(np.transpose(ww0),Rx),ww0)

cp = plt.contour(X, Y, J, levels)
plt.title('insert title')
plt.xlabel('w0')
plt.ylabel('w1')
plt.xlim(wiener0 - scale_factor,wiener0 + scale_factor, n_size)
plt.ylim(wiener1 - scale_factor, wiener1 + scale_factor, n_size)
plt.plot(w_0,w_1)
plt.show()

    

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        