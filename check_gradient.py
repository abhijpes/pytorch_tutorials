import numpy as np
import torch
from torch.autograd import Variable
import torch.nn
x = Variable(torch.Tensor(np.linspace(-2,1,100)), requires_grad = True)
y = x**2 + 2*x + 4
target = -1
index = np.where(x.data.numpy() >= -1)[0][0]
print(index)
