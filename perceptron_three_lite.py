import numpy as np
from numpy import random,dot
import time
start_time = time.time()
weight = 2*random.random((4,4))-1
w1,w2,w3,w4 = 5,4,1,1
x1,x2,x3,x4 = 0,1,0,0
a = 0.3
def logist(net):
 return 1/(1+np.exp(-a*net))
summator = w1*x1 + w2*x2 + w3*x3 + w4*x4
out_neu = logist(summator)
print(out_neu)
inp = np.array([[1],[1],[1],[1]])
tout = np.array([[0.81757],[0.76852],[0.57444],[0.57444]]).T
for i in range(1000):
 out = 1/(1+np.exp(-a*weight))
 weight += dot(inp.T,(tout-out)*out*(1-out))
print(weight)
weight1 = (weight[0][0] + weight[1][0] + weight[2][0] + weight[3][0])/4
weight2 = (weight[0][1] + weight[1][1] + weight[2][1] + weight[3][1])/4
weight3 = (weight[0][2] + weight[1][2] + weight[2][2] + weight[3][2])/4
weight4 = (weight[0][3] + weight[1][3] + weight[2][3] + weight[3][3])/4
print(time.time()-start_time)
while True:
 train_inp = np.array([[int(input())],[int(input())],[int(input())],[int(input())]])
 train_sum = weight1*train_inp[0] + weight2*train_inp[1] + weight3*train_inp[2] + weight4*train_inp[3]
 out_test = 1/(1+np.exp(-a*train_sum))
 print(out_test)
