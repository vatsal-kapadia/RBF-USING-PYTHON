import numpy as np 
import matplotlib.pyplot as plt 
def Gaussian(center,data_point): 
 sigma = 4 
 return np.exp(-np.linalg.norm((center-data_point)/sigma)**2) 
def Centers(data): 
 M = np.zeros([4,4]) 
 N = data.shape[0] 
 for i in range(N): 
  for j in range(N): 
   M[i,j] = Gaussian(np.array(data[i,0]),np.array(data[j,0]))  
  return M 
  
def weights(M,Y): 
 weights = np.dot(np.linalg.pinv(M), Y) 
 return weights 

def step(X): 
 if X>0: 
  return 1 
 else: 
  return 0

def predict(M,weights): 
 predictions = np.dot(M, weights) 
 l_predict = [] 
 for i in range(M.shape[0]): 
  l_predict.insert(i,step(predictions[i])) 
 print(l_predict) 
 return np.array(l_predict) 

def fit(data): 
 M = Centers(data) 
 weight = weights(M,data[:,1]) 
 return M, weight 

data = np.array([[[1, 1], 1],[[1, 0], 1],[[0, 1], 1],[[0, 0], 0]]) 
x = data[:,0] 
y = data[:,1] 
M, weight = fit(data) 
y_pred = predict(M,weight)
