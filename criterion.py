import numpy as np
from activation_function import Softmax


class CrossEntropyLoss():
    '''Categorical Cross-Entropy loss'''

    def __init__(self):
        self.softmax = Softmax()
        self.cache = None
        self.true = None

    def _forward(self, Y_pred, Y_true):
        self.true = Y_true
        prob = self.softmax._forward(Y_pred)
        self.cache = prob 
        loss = -np.sum(np.multiply(Y_true,np.log(prob, where=prob>0)))
        return loss

    def _backward(self):
        # dX = pred-label
        prob = self.cache
        up_grad = prob - self.true
        return up_grad


class SGDMomentum():
    def __init__(self, params, lr=0.001, momentum=0.99, reg=0):
        self.l = len(params)
        self.parameters = params
        self.velocities = []
        for param in self.parameters:
            self.velocities.append(np.zeros(param['wgt'].shape))
        self.lr = lr
        self.rho = momentum
        self.reg = reg

    def step(self):
        for i in range(self.l):
            self.velocities[i] = self.rho*self.velocities[i] + (1-self.rho)*self.parameters[i]['grad']
            self.parameters[i]['wgt'] -= (self.lr*self.velocities[i] + self.reg*self.parameters[i]['wgt'])