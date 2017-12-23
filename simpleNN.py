import torch
import numpy
from torch.autograd import Variable
import torch.nn as nn

#nn.Module contains the layers of a network and also the methods for the forward and the backward pass - It acts as the Base of the neural network
#Parameter class - a subclass of the Variable class which acts as the input as the derivitive is not expected wrt this variable.
# If defined they show up in the call to the iterator parameters()

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
    def forward_pass(self):

    def backward_pass(self):
