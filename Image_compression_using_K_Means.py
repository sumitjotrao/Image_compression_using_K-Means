from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import math
import statistics as st
import random



data2=loadmat(r"C:\Mydata\ML Course\machine-learning-ex7\ex7\bird_small.mat")

A=data2['A']
A=A/255
#print(A)
# #print(data2)
# X=data2['X']
# m=X.shape[0]
#
K = 16

randomlist=random.sample(range(1,len(A)),K)

initial_centroids = [A[i][i] for i in randomlist]
X=[]

for i in range(len(A)):
    for j in range(len(A[i])):
        X.append(A[i][j])
m=len(X)


# centroid_list=[]




def closestcentroid(list,X,initial_centroids):

    for i in range(m):
        temp_list = []
        for j in range(K):


           temp_list.append(math.sqrt(((X[i][0]-initial_centroids[j][0])*(X[i][0]-initial_centroids[j][0]))+((X[i][1]-initial_centroids[j][1])*(X[i][1]-initial_centroids[j][1]))+((X[i][2]-initial_centroids[j][2])*(X[i][2]-initial_centroids[j][2]))))

        min_index=temp_list.index(min(temp_list))
        list.append(min_index)
        del temp_list
    return list

def computecentroidmeans(list,X):
    mean_dt=[]
    for j in range(K):
       temp1 = []
       for i in range(len(list)):
           if list[i]==j:
              temp1.append(X[i])
       a_x=[]
       a_y=[]
       a_z=[]
       for k in range(len(temp1)):
           a_x.append(temp1[k][0])
           a_y.append(temp1[k][1])
           a_z.append(temp1[k][2])
       m_x=st.mean(a_x)
       m_y=st.mean(a_y)
       m_z=st.mean(a_z)
       mean_dt.append([m_x,m_y,m_z])

       del temp1
       del a_x
       del a_y
    return mean_dt


#
max_iter=10


#
#
# x1=[]
# x2=[]
# for i in range(m):
#     x1.append(X[i][0])
#     x2.append(X[i][1])

def runkmeans(X,initial_centroids):
 s=0
 while s<max_iter:
  list = []
  list1=closestcentroid(list,X,initial_centroids)

  initial_centroids=computecentroidmeans(list1,X)
  #print(initial_centroids)

  s=s+1
 return initial_centroids,list1

a,b=runkmeans(X,initial_centroids)

print(b)

X_new=[]
for i in b:
    X_new.append(initial_centroids[i])

X=np.array(X)
X_new=np.array(X_new)
print(X.shape)
print(X_new.shape)
x_new_mat=np.reshape(X_new,(128,128,3))

print(x_new_mat)



from skimage.color import rgb2lab, rgb2gray, lab2rgb
from skimage.io import imread, imshow
image_gs = imread(r"C:\Mydata\ML Course\machine-learning-ex7\ex7\bird_small.png")
#print(image_gs)
#fig, ax = plt.subplots(figsize=(9, 16))
fig=plt.figure()
ax1=fig.add_subplot(2,2,1)
ax2=fig.add_subplot(2,2,2)

imshow(image_gs, ax=ax1)
imshow(x_new_mat, ax=ax2)
plt.show()