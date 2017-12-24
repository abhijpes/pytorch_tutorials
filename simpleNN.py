import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn

#nn.Module contains the layers of a network and also the methods for the forward and the backward pass - It acts as the Base of the neural network
#A Module is roughly equivalent to the leyrs in a neural network. Have functions defined to compute the output given the input.
#Module.forward() - Is a function thaty has to be overriden by the subclass
#Parameter class - a subclass of the Variable class which acts as the input as the derivitive is not expected wrt this variable.
#If defined they show up in the call to the iterator parameters()
'''N, D_i, D_out =  20, 1000, 10
inp = Variable(torch.randn(N, D_in))
opt = Variable(torch.randn(N, D_out))
torch.nn.Sequential(torch.nn.Linear(D_in, H))
'''
# The linear layer - takes in the input and ouput dimentions. As a result, generates a weight matrix of dimensions(out X in).
# To define a layer, linear1 = torch.nn.Linear(Inp, H) -Implementation of the linear layer - inp.matmul(weight.t()) + bias (if defined)
# To define a sequenc of op
#=============================================
#Feedforward Layer Numpy
#=============================================
class Numpy_net():
    def __init_(D_in, N, H, D_out, rate = 1e-6):
        self.N, self.D_in, self.H, self.D_out = N, D_in, H, D_out
        self._D_inH = np.random.randn(self.D_in, self.H)
        self._HD_out = np.random.randn(self.H, self.D_out)
        x = np.random.randn(N, D_in)
        y = np.random.randn(N, D_out)
        self._rate = rate if rate is not None
    def  forward():
        # Define what the forward pass for the network is
        h = x.dot(self._D_inH)
        h_relu = np.maximum(h, 0)
        y_out = h_relu.dot(self._HD_out)
        # Question - What's the dimension of the vector y_out, h_relu, h?
    def backprop():
        # Implemention of the MSE
        cost = np.square(y-y_out).sum()
        print("Iteration - {0} Loss - {1}".format(i+1, cost))
        gradient_y = 2.0 *(y - y_out)
        grad_w2 = h_relu.T.dot(grad_y_pred)
        grad_h_relu = grad_y_pred.dot(w2.T)
        grad_h = grad_h_relu.copy()
        grad_h[h < 0] = 0
        grad_w1 = x.T.dot(grad_h)

        #Updating the weights now..
        w1 = w1 - self._rate * grad_w1
        w2 = w2 - self._rate * grad_w2
#=============================================
