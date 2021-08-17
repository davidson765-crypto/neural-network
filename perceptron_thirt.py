import numpy as np
from numpy import random,dot


weight = 2*random.random((4,1))-1

w1 = 5
w2 = 4
w3 = 1
w4 = 1

x1 = 0
x2 = 0
x3 = 0
x4 = 1




a = 0.3

def logist(net):
	return 1/(1+np.exp(-a*net))

summator = w1*x1 + w2*x2 + w3*x3 + w4*x4

out_neu = logist(summator)
print(out_neu)

inp = np.array([[1],[0],[0],[0]])
tout = np.array([[0.81757]]).T
for i in range(1000):
	out = 1/(1+np.exp(-a*weight))
	weight += dot(inp.T,(tout-out)*out*(1-out))

print(weight[0])

weights1 = np.array([[weight[0]]])




inp = np.array([[0],[1],[0],[0]])
tout = np.array([[0.76852]]).T
for i in range(1000):
	out = 1/(1+np.exp(-a*weight))
	weight += dot(inp.T,(tout-out)*out*(1-out))

print(weight[1])

weights2 = np.array([[weight[1]]])


inp = np.array([[0],[0],[1],[0]])
tout = np.array([[0.57444]]).T
for i in range(1000):
	out = 1/(1+np.exp(-a*weight))
	weight += dot(inp.T,(tout-out)*out*(1-out))

print(weight[2])

weights3 = np.array([[weight[2]]])



inp = np.array([[0],[0],[0],[1]])
tout = np.array([[0.57444]]).T
for i in range(1000):
	out = 1/(1+np.exp(-a*weight))
	weight += dot(inp.T,(tout-out)*out*(1-out))

print(weight[3])

weights4 = np.array([[weight[3]]])




weights = np.concatenate((weights1,weights2,weights3,weights4))

print(weights)


train_inp = np.array([[0],[0],[1],[1]])


train_sum = weights[0]*train_inp[0] + weights[1]*train_inp[1] + weights[2]*train_inp[2] + weights[3]*train_inp[3]


out_test = 1/(1+np.exp(-a*train_sum))

print(out_test)