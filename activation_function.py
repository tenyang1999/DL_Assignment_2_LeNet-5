import numpy as np


class ReLU():
    def __init__(self):
        self.cache = None

    def _forward(self, X):
        out = np.maximum(0,X)
        self.cache = X
        return out

    def _backward(self,up_grad):
        X = self.cache
        dX = up_grad
        dX[X <= 0] = 0
        return dX


class Sigmoid():
    def __init__(self):
        self.cache = None

    def _forward(self, X):
        self.cache = X
        # 優化使得原始的Sigmoid 會有overflow warning
        X_ravel = X.ravel()
        length = len(X_ravel)
        y = []
        for index in range(length):
            if X_ravel[index] >= 0:
                y.append(1.0 / (1 + np.exp(-X_ravel[index])))
            else:
                y.append(np.exp(X_ravel[index]) / (np.exp(X_ravel[index]) + 1))
        return np.array(y).reshape(X.shape)

    def _backward(self,up_grad):
        X = self.cache
        X = X.astype(np.float64)
        dX = up_grad*self._forward(X)*(1-self._forward(X))
        return dX


class Softmax():
    def __init__(self):
        self.cache = None

    def _forward(self, X):
        # for solve the problem of operation overflow, minus max in each proba
        max_ = np.max(X, axis=1).reshape(X.shape[0],1)
        Y = np.exp(X-max_)
        Z = Y / np.sum(Y, axis=1,keepdims=True)
        self.cache = (X, Y, Z)
        return Z  # distribution

    def _backward(self, up_grad):
        X, Y, Z = self.cache
        dZ = np.zeros(X.shape)
        dY = np.zeros(X.shape)
        dX = np.zeros(X.shape)
        N = X.shape[0]
        for n in range(N):
            i = np.argmax(Z[n])
            dZ[n,:] = np.diag(Z[n]) - np.outer(Z[n],Z[n])
            M = np.zeros((N,N))
            M[:,i] = 1
            dY[n,:] = np.eye(N) - M
        dX = np.dot(up_grad,dZ)
        dX = np.dot(dX,dY)
        return dX
