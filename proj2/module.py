from torch import FloatTensor
import torch
import math
import numpy as np

class Module(object):
    def forward(self, input):
        raise NotImplementedError
    def backward(self, upstream_derivative):
        raise NotImplementedError
    def param(self):
        return []
    
class Linear(Module):
    '''
    Assumption based:
        Linear Equation : Y = X * W + b.
        Data Structure  : Rows represent data, colums represent features.   
    '''
    def __init__(self, input_dim, output_dim, bias=True, initOption='Normal'):
        super(Linear).__init__()
        self.name = 'Linear'
        
        self.input_dim, self.output_dim = input_dim, output_dim
        
        
        self.w = FloatTensor(input_dim, output_dim)
        self.gradW = FloatTensor(input_dim, output_dim)
        
        self.b = FloatTensor(output_dim)
        self.gradB = FloatTensor(output_dim)
        
        if bias:
            self.b = FloatTensor(output_dim)
            self.gradB = FloatTensor(output_dim)
        else:
            self.b = None
            self.gradB = None
        
        self.initOption = initOption
        self.initParameters()
        
    def initParameters(self):
        '''
        Different methods for parameter initialization.
        '''
        if self.initOption == 'Normal':
            self.w.normal_()
        if self.initOption == 'Zero': 
            self.w.zero_()
        if self.initOption == 'He': 
            # 'He initialization' recommends for layers with a ReLU activation
            self.w.normal_().mul_(math.sqrt(2/(self.input_dim)))
        if self.initOption == 'Xavier':
            # 'Xavier initialization' recommends for layers with a tanh activation
            self.w.normal_().mul_(math.sqrt(2/(self.input_dim + self.output_dim)))
            
        self.gradW.fill_(0)
        
        if self.b is not None:
            self.b.normal_()
            self.gradB.fill_(0)
            
    def forward(self, input):
        '''
        Forward Pass: 
            Y = X * W + b.
        '''
        self.input = input
    
        if self.b is not None:
            self.output = self.input.matmul(self.w).add(self.b) # Broadcast
        else:
            self.output = self.input.matmul(self.w)
        return self.output
         
    def backward(self, gradwrtoutput): 
        '''
        Backpropagation: gradwrtoutput = batch_size * output_dim
            dW = X^T * dL/dY
            db = (dL/dY)^T * I
            dX = dL/dY * W^T.
        '''
        self.gradW.add_(self.input.t().matmul(gradwrtoutput))
        
        if self.b is not None:
            self.gradB.add_(gradwrtoutput.sum(0))
        return gradwrtoutput.matmul(self.w.t())
    
    def zero_grad(self):
        '''
        Set gradient to 0.
        '''
        self.gradW.zero_()
        
        if self.b is not None:
            self.gradB.zero_()
            
    def param(self):
        '''
        Return parameters.
        '''
        if self.b is not None:
            return [(self.w, self.gradW),
                    (self.b, self.gradB)]
        else:
            return [(self.w, self.gradW)]
        
class ReLU(Module):
    def __init__(self):
        self.name = 'ReLU'
        
    def forward(self, input):
        self.input = input
        self.positive = (input > 0).float()
        self.output = self.input.mul(self.positive)
        return self.output
    
    def backward(self, upstream_derivative):
        '''
        f(x) = 0 if  x < 0
               1 if  x > 0
        '''
        return self.positive.mul(upstream_derivative)
    
    def param(self):
        return [(None, None)]
    
class Tanh(Module):
    def __init__(self):
        self.name = 'Tanh'

    def forward(self, input):
        self.input = input
        self.output = torch.tanh(input)
        return self.output
    
    def backward(self, upstream_derivative):
        return upstream_derivative * (torch.ones_like(self.input) - self.output**2)
    
    def param(self):
        return [(None, None)]

class sigmoid(Module):
    def __init__(self):
        self.name = 'Sigmoid'

    def forward(self, input):
        self.input = input
        self.output = 1 / (1 + (-input).exp())
        return self.output
    
    def backward(self, upstream_derivative):
        return upstream_derivative * (self.output - self.output**2)
    
    def param(self):
        return [(None, None)]

class BN(Module):
    
    def __init__(self):
        self.name = 'BN'

    def forward(self, input):
        epsilon = 1e-6
        self.input = input
        sum = input.mean(dim = 0)
        std = input.std(dim = 0)
        denominator = torch.sqrt(std**2 + epsilon)
        self.output = (input - sum) / denominator
        return self.output
    
    def backward(self, upstream_derivative):
        return upstream_derivative
    
    def param(self):
        return [(None, None)]


class MLP(Module):
    '''
    Multiple layer perceptrons.
    '''
    def __init__(self, *args):
        super(MLP).__init__()
        
        self.sequential_modules = []
        for module in args:
            self.sequential_modules.append(module)
    
    def __call__(self, input):
        self.forward(input)
        return self.output
            
    def forward(self, input):
        self.input = input
        output = input
        for module in self.sequential_modules:
            output = module.forward(output)
        self.output = output
        return self.output
    
    def backward(self, upstream_derivative):
        for module in reversed(self.sequential_modules):
            upstream_derivative = module.backward(upstream_derivative)
        self.grad = upstream_derivative
        return self.grad
    
    def param(self):
        '''
        Parameters of sequential model are combined by children modules.
        '''
        parameters = []
        for module in self.sequential_modules:
            parameters.extend(module.param())
        return parameters
    
    def zero_grad(self):
        '''
        Set gradient to 0.
        '''
        for tup in self.param():
            weight, grad = tup
            if (weight is None) or (grad is None):
                continue
            else:
                grad.zero_()
                
    def add(self, *args):
        '''
        Add modules to sequential model.
        '''
        for module in args:
            self.sequential_modules.append(module)        

    def initParameters(self):
        for module in self.sequential_modules:
            if module.name == 'Linear':
                module.initParameters()
            