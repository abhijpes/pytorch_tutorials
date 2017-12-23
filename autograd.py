

# variable in autograd
# a Variable packs a tnesors inside and holds  the methods defined by that tensor.
#To compute all the gradients after performing all the forward propagation, .backward() can be used to automatically compute the gradients
#autograd.Variable
# extracting the tensor value form the variable - autograd.Variable.data
# to extract the gradients, .grad will of the help

#--------------------------------
# Functions in autograd -
#Variables and functions do make up the acyclic graphs..
# Every variable that was created due to some tranformations and operations can refer to the function that created that variable using .grad_fn()
from torch.autograd import Variable
import numpy
x = Variable(torch.ones(2,2))
# probably can also create a variable fircetly using the numpy array too!!
x = Variable(torch.Tensor(numpy.linspace(1,10,30)), requires_grad = True)
# Performing an operation on the varaible
x.add_(2)
# Now the function that has updated the variable x will be updated, pointed to by the parameter, grad_fn
print(x.grad_fn)
# Some more computations
z = y*y*3
out = z.mean()
prrint(z, out)
# gradients in autograd
out.backward() # performs the different the variable "out" wrt x (the variable we started the tranformation from)
print(out)
#----------------------

x = torch.randn(3)
x = Variable(x, requires_grad = True)
y = x*2
while(y.data.norm() < 1000):
    y = y*2
gradients =
#-------------
#Creator property for a variable  - Tells what was the last operation that was done to generate the variable.
z = x + y
z.creator() # prints the ops that where used to define z
