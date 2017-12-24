import torch
import torchvision
import numpy

# Constructing a 5x3 matrix in torch
x=torch.Tensor(5, 3)
print(x)

# Printing the siuze of this metrix
print(x.size()) # which is of the type tuple
print("The type of the size matrix is:")
print(type(x.size))

#initializing a random matrix of a particular dimension
x = torch.rand(5,3)
print("A randomly initialised matrix of the type 5,3")
print(x)

#initialising another matrix of similar dimension

y= torch.rand(5,3)
print("Below is the matrix y")
print(y)

# Adding 2 vectors
print(x+y)
#Above is demonstrated using operator overloading
# Alternative syntaxes
# torch.add(x,y,out=result)
# result = y.add(x)
# add inplace , add the 3 tensors and stor the result in y itself
# y.add_(x)
print("Demonstation of adding inplace")
print(y.add_(x))
#Converting from torch array to a numpy array is also easy AF

arr = y.numpy()
print(arr)
# Like numpy, torch can perform appropriate sizing too
tmparr = torch.ones(5)
print(tmparr)
print("Now adding 10 to all the elements in the array")
print(tmparr.add_(10))
#likewisem converting from numpy array to torch array is also easy AF
tmparr = numpy.ones(5)
tmparr = torch.from_numpy(tmparr)
# A torch array and a numpy array point to the same underlying memory though
#A change in the value for the underlyting numpy array will be reflected in torch too..
numpy.add(tmparr,  1, out = res)
print(res)
# Last but not the least, if you might want to hard code a vector an define using variable datatype
x = torch.FloatTensor([1.12,2.34123,3.3546])
print(x)
x= torch.LongTensor([1231342134,234523,235213])
#Or the best way to do is from numpy
# To select whether the  tensor need s to be defined on the GPU or the CPU: Use the .cpu() or .cuda()
#===================================
# Random sampling of data(inplace)
print(torch.Tensor.uniform_(2,6))
seed = 10
tmp = torch.Tensor.normal(seed)
print(tmp)
#==========================
# matrix multiplication  - using the matmul function
tmparr.matmul(x.t()) # where the t()_ represents the transpose operation.
#=================
# Performing a ReLU is just limiting the input values above a certain threshold.
# An h.clamp(min=0) will do the job
tmp.clamp(min=0)
print(tmp)
