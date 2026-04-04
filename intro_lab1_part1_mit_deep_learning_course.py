import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

#pytorch - machine learning library, provides an interface for creating and manipulating tensors(data structures can think of it as multi dimensional arrays)
#tensors - n-dimensional arrays of base datatypes(string, integers, float, etc) -- they provide a way to generalize vectors and matrices to higher dimensions
#pytorch - provides the ability to perform computation on these tensors, define neural networks and train the efficiently

#int and float tensors of dimension 0
integer = torch.tensor(123)
float = torch.tensor(3.1415)
print(integer.ndim, float.ndim)

#tensors of dimension 1
fibonacci = torch.tensor([1,1,2,3,5,8])
count_to_50 = torch.tensor(range(50))
print(fibonacci.ndim, count_to_50.ndim)

#NOTE : image processing and computer vision uses a 4th dimension tensor corresponding to batch size, number of color channels, image height and image width

#2d tensor [[]] = 2d  [[[]]] = 3d and so on
matrix = torch.tensor([[]])
matrix = torch.zeros(10,10)
print(matrix)

#4d tensor with size 10*3*256*256
print("image var")
image = torch.tensor([[[[]]]])
image = torch.zeros(10,3,256,256)
print(image.ndim, image)


#shape - provides the nr of elems in each tensor dimension -- use slicing to acces subtensors within a higher rank tensor
row_vector = matrix[1]
column_vector = matrix[:,1]
scalar = matrix[0,1]


#Computations on Tensors

#computations are handled by the computation graph the parent defines the sum between the children -- in our case c and d could be parents and a and b children
#parent c = a+b and parent d = a*b

a = torch.tensor(10)
b = torch.tensor(20)

c = torch.add(a, b)
d = a * b

#Neural Networks in Pytorch

#torch.nn.Module - base class for all neural network modules in pytorch (provides a framework for building and training neural networks)
#define a perceptron define by just one dense = fully connected = linear
# y = sigma_func(W * x + b) - where W=matrix of weights, b=bias, x=input, sigmoid activation function, y=output
#we can use the computation graph W, x, b child nodes and z = x * W + b parent of this nodes and out=sigmoid_func(z) parent of z node

class custom_dense_layer(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):            #num of input nodes and num of output nodes and x input to the layer
        super(custom_dense_layer, self).__init__()
        self.W = torch.nn.Parameter(torch.randn(num_inputs, num_outputs))
        self.bias = torch.nn.Parameter(torch.randn(num_outputs))

    def forward(self, x):
        z = torch.matmul(x * self.W, self.bias)
        y = torch.sigmoid(z)
        return y

#define a layer and test the output

num_inputs = 1
num_outputs = 2
layer = custom_dense_layer(num_inputs, num_outputs)
x_input = torch.tensor([[1, 2.]])
y = layer(x_input)

print(f"input shape: {x_input.shape}")
print(f"output shape: {y.shape}")
print(f"output result: {y}")

#nr of nn.Modules that are commonly used in neural networks - nn.Linear or nn.Sigmoid modules
#instead of using a single Module to define our neural network we will use the nn.Sequential module and a single nn.Linear layer to define our network 
#Sequential API - create neural networks by statcking together layers like building blocks

n_input_nodes = 2
n_output_nodes = 3
model = nn.Sequential(nn.Linear(n_input_nodes, n_output_nodes), nn.Sigmoid())
x_input = torch.tensor([[1,2.]])
model_output = model(x_input)
print(x_input.shape, y.shape, y)

#we can create more flexible models by subclassing nn.Module
#nn.Module - allows us to group layers together flexibly to define new arhitectures
#previosuly - with custom_dense_layer we can subclass nn.Modules to create a class for our model and then define the forward pass through the network using the forward function
#subclassing - flexibility to define custom layers, custom training loops, custom activation functions and custom models

class linear_with_sigmoid_activation(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(linear_with_sigmoid_activation ,self).__init__()

        self.linear = nn.Linear(num_inputs, num_outputs)
        self.activation = nn.Sigmoid()

    def forward(self, inputs):
        linear_output = self.linear(inputs)
        output = self.activation(linear_output)
        return output
    
n_input_nodes = 2
n_output_nodes = 3
model = linear_with_sigmoid_activation(n_input_nodes, n_output_nodes)
x_input = torch.tensor([[1, 2.]])
y = model(x_input)
print(x_input.shape, y.shape, y)

#nn.Module - offers a lot of flexibility to define custom models. we can use boolean arguments in the forward function to specify different network behaviours 
#for ex different behaviors during training and inference 
#define a boolean argument to control this behaviors

class linear_but_sometimes_identity(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(linear_but_sometimes_identity, self).__init__()

        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, inputs, identity = False):
        if identity:
            return inputs
        else:
            return self.linear(inputs)

#test the identity models

model = linear_but_sometimes_identity(num_inputs=2, num_outputs=3)
x_input = torch.tensor([[1, 2.]])
out_with_linear = model(x_input)
out_with_identity = model(x_input, identity = True)
print(f"input: {x_input}")
print("Network linear output: {}; network identity output: {}".format(out_with_linear, out_with_identity))

#Automatic Differentiation in Pytorch

#torch.autograd - used for automatic differentiation which is critical for training deep learning models with backpropagation
# .backforward() - method to trace operations for computing gradients
#When a forward pass is made through the network, PyTorch builds a computational graph dynamically; then, to compute the gradient, the .backward() method is called to perform backpropagation

#gradient computation
#y = x^2

x = torch.tensor(3.0, requires_grad=True)   #requires grad control wether autograd should record operations on that tensor
y = x**2
y.backward()
dy_dx = x.grad
print(dy_dx)


#In training neural networks, we use differentiation and stochastic gradient descent (SGD) to optimize a loss function.
#autograd can be used to compute and access derivatives

#function minimization with autograd and gradient descent

x = torch.randn(1)
learning_rate = 1e-2
history = []
x_target_val = 4

for i in range(500):
    x = torch.tensor([x], requires_grad=True)
    loss = (x - x_target_val) **2
    loss.backward()
    x = x.item() - learning_rate * x.grad   
    history.append(x.item())

plt.plot(history)
plt.plot([0, 500], [x_target_val, x_target_val])
plt.legend(('Predicted', 'True'))
plt.xlabel('Iteration')
plt.ylabel('x value')
plt.show()