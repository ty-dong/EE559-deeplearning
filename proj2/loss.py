from torch import FloatTensor
import torch
import math
import numpy as np

class Loss_fnc(object):
    def loss(self, output, target):
        raise NotImplementedError

    def backward(self, MLP):
        raise NotImplementedError
        
class Loss_fnc(object):
    def forward(self, output, target):
        raise NotImplementedError

    def backward(self, MLP):
        raise NotImplementedError

class MSE(Loss_fnc):
    '''
    Mean squared loss.
    '''
    def __init__(self, model):
        super(MSE).__init__()
        self.diff = 0
        self.model = model
        
    def __call__(self, output, target):
        self.forward(output, target)
        return self
          
    def forward(self, output, target):
        '''
        Calculate the error.
        '''
        self.output, self.target = output, target
        self.diff = self.output - self.target
        length = self.diff.size()[0]
        self.loss = self.diff.pow(2).sum() / length
        return self.loss
    
    def backward(self):
        '''
        Gradient of loss.
        '''
        length = self.diff.size()[0]
        back = 2 * self.diff / length
        
        return self.model.backward(back)
    
class CrossEntropy(Loss_fnc):
    '''
    Cross Entropy loss.
    '''
    def __init__(self, model):
        super(MSE).__init__()
        self.model = model
        
    def __call__(self, output, target):
        self.forward(output, target)
        return self
          
    def forward(self, output, target):
        '''
        Calculate the error.
        '''
        # output_size : batch_size * output_dim
        self.output = output
        self.target = target
        softmax = torch.softmax(output, dim = 1)
        loss = (-target * (softmax.log())).sum(dim = 1)
        self.loss = loss.mean()
        return self.loss
    
    def backward(self):
        '''
        Gradient of loss.
        '''
        n = self.output.size()[0]
        softmax = torch.softmax(self.output, dim = 1)
        y = -self.target[:, 0] * softmax[:, 1] + self.target[:, 1] * softmax[:, 0]
        y = y.view(-1, 1)
        grad = torch.cat([y, -y], dim = 1) / n
        return self.model.backward(grad)
    