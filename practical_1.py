#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Load the Pandas libraries with alias 'pd' 
import pandas as pd 
# Read data from file 'filename.csv' 
# (in the same directory that your python process is based)
# Control delimiters, rows, column names with read_csv (see later) 
data = pd.read_csv("assignment1_data.csv") 
# Preview the first 5 lines of the loaded data 
#data.head()
print(data.shape)


# In[4]:


import numpy as np 
x = data.iloc[:,0]
y = data.iloc[:,1]
print(x.shape)
print(y.shape)

x = x.to_numpy()
y = y.to_numpy()

    
#print(prediction.shape)


# In[63]:


N = len(x)
alpha = 0.001
R_x = np.array([[5,-1,-2],[-1,5,-1],[-2,-1,5]])
R_x_inv = np.linalg.inv(R_x)
r_yx = np.array([[1],[5.3],[-3.9]])
w = np.zeros((3,1))
w_history = np.zeros((N,3))
print(w_history.shape)
for p in range(N):
    #MMSE 
    w = w + 2*alpha*(r_yx-np.dot(R_x,w))
    # Newton: w = w + 2*alpha*np.dot(R_x_inv,(r_yx-np.dot(R_x,w)))
    w_history[p,:] = w.ravel()
print (w)


# In[64]:


import matplotlib.pyplot as plt 
w0 = w_history[:,0]
w1 = w_history[:,1]
w2 = w_history[:,2]
iterations = np.arange(0,N)
plt.plot(iterations,w_history)
plt.xlabel('iterations')
plt.ylabel('weights')
plt.title('weight convergence with MMSE')
plt.show()


# In[65]:


plt.plot(w0,w1)
plt.show()


# In[70]:


J_min = 0 
x0 = w_history[N-1,0]*np.ones((N,1))
x1 = w_history[N-1,1]*np.ones((N,1))
x2 = -0.5*np.ones((N,1))
w_o = np.concatenate((x0,x1,x2),axis=1)

p = w_history-w_o
p_t = np.transpose(p)
J = np.dot(np.dot(p,R_x),p_t)
print(J.shape)


# In[79]:


X1,X2 = np.meshgrid(w0,w1)
#cp = plt.contour(X1, X2, J)
levels = [0.0, 0.01, 0.1 ,1.0 ,4.0]
#cp = plt.contour(X1, X2, J, colors='black', linestyles='dashed', linewidths=1)

cp = plt.contour(X1, X2, J, levels)
plt.clabel(cp, inline=1, fontsize=10)
plt.xlabel('w0')
plt.ylabel('w1')
plt.plot(w0,w1)
plt.show()


# In[83]:


#calculate J_min 
E_y_sq = np.mean(np.square(y))
print(E_y_sq)
J_min = E_y_sq-np.dot(np.dot(np.transpose(r_yx),R_x_inv),r_yx)
print(J_min)


# In[86]:


J_min = np.amin(J)
print(J_min)


# In[ ]:




