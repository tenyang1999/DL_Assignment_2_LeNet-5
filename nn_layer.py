import numpy as np


class FC():
    def __init__(self,D_in,D_out):
        self.cache = None
        self.W = {"wgt":np.random.uniform(-1,1,(D_in,D_out)),'grad':0}
        self.b = {'wgt':np.random.uniform(-1,1,(D_out)),"grad":0}

    def _forward(self, X):
        self.cache = X
        out = np.dot(X,self.W['wgt'])+self.b['wgt']
        return out

    def _backward(self,up_grad):
        X = self.cache
        dX = np.dot(up_grad, self.W['wgt'].T)
        self.W['grad'] = np.dot(X.T,up_grad)
        self.b['grad'] = np.sum(1*up_grad, axis=0)
        self._update_params()
        return dX

    def _update_params(self,lr=0.001):
        self.W['wgt'] -= self.W['grad']*lr
        self.b['wgt'] -= self.b['grad']*lr


class Conv():
    def __init__(self, C_in, C_out, kernel, stride=1, padding=0, bias=True):
        '''
        Parameters
        ------------
        C_in : int
            Number of input neuron
        C_out : int
            Number of output neuron
        Kernel : int
            Kernel size is [kernel,kernel]
        Stride : int
            determines how many pixels the kernel moves each time
        pad : int
            adds extra rows and columns of pixels to the edges

        '''
        self.C_in = C_in
        self.C_out = C_out
        self.Kernel = kernel
        self.Stride = stride
        self.pad = padding
        self.W = {'wgt': np.random.uniform(0.0, 1.0, (C_out,C_in, kernel, kernel)), 'grad': 0}
        self.b = {'wgt': np.random.uniform(0.0, 1.0,C_out), 'grad': 0}
        self.cache = None

    def _forward(self, X):
        '''
        Parameters
        ------------
        N : int
            Number of train examples
        C_in : int
            Number of input neuron
        H : int
            Height of the image
        W : int
            Weight of the image
        '''
        self.cache = X
        # add padding for image
        X = np.pad(X, ((0,0),(0,0),(self.pad,self.pad),(self.pad,self.pad)), 'constant')
        (N, C_in, H, W) = X.shape
        # height & weight after convolution
        new_H = int(np.floor((H + 2*self.pad - self.Kernel)/self.Stride)+1)
        new_W = int(np.floor((W + 2*self.pad - self.Kernel)/self.Stride)+1)
        # zero Y

        Y = np.zeros((N, self.C_out, new_H, new_W))

        for n in range(N):
            for c in range(self.C_out):
                for h in range(new_H):
                    for w in range(new_W):
                        Y[n, c, h, w] = np.sum(X[n, :, h*self.Stride:h*self.Stride+self.Kernel, 
                                                       w*self.Stride:w*self.Stride+self.Kernel] * self.W['wgt'][c, :, :, :]) + self.b['wgt'][c]

        return Y

    def _backward(self, up_grad):
        # up_grad (N, C_out, new_H, new_W)
        # W (C_out, C_in, Kernel, Kernel)
        X = self.cache
        (N, C_in, H, W) = X.shape
        new_H = int(np.floor((H + 2*self.pad - self.Kernel)/self.Stride)+1)
        new_W = int(np.floor((W + 2*self.pad - self.Kernel)/self.Stride)+1)

        W_rot = np.rot90(np.rot90(self.W['wgt']))

        dX = np.zeros(X.shape)
        dW = np.zeros(self.W['wgt'].shape)
        db = np.zeros(self.b['wgt'].shape)

        # dW
        for co in range(self.C_out):
            for ci in range(C_in):
                for h in range(self.Kernel):
                    for w in range(self.Kernel):
                        dW[co, ci, h, w] = np.sum(X[:,ci,h:h+new_H,w:w+new_W] * up_grad[:,co,:,:])

        # db
        for co in range(self.C_out):
            db[co] = np.sum(up_grad[:,co,:,:])

        up_grad_pad = np.pad(up_grad, ((0,0),(0,0),(self.Kernel,self.Kernel),(self.Kernel,self.Kernel)), 'constant')

        # dX
        for n in range(N):
            for ci in range(C_in):
                for h in range(H):
                    for w in range(W):
                        dX[n, ci, h, w] = np.sum(W_rot[:,ci,:,:] * up_grad_pad[n, :, h:h+self.Kernel,w:w+self.Kernel])

        return dX


class MaxPool():
    def __init__(self, kernel, stride):
        self.Kernel = kernel
        self.Stride = stride
        self.cache = None
        self.ori_shape = None

    def _forward(self, X):

        N, C_in, H, W = X.shape
        # height & weight after convolution
        if H % 2 == 1:
            self.ori_shape = True
            X = np.pad(X, ((0,0),(0,0),(0,1),(0,1)), 'constant')
            N, C_in, H, W = X.shape

        new_H = int(np.floor((H - self.Kernel)/self.Stride)+1)
        new_W = int(np.floor((W - self.Kernel)/self.Stride)+1)
        # zero array for the next layer
        Y = np.zeros((N, C_in, new_H, new_W))
        M = np.zeros(X.shape) # mask

        for n in range(N):
            for c in range(C_in):
                for h in range(new_H):
                    for w in range(new_W):
                        Y[n, c, h, w] = np.max(X[n, c, h*self.Stride:h*self.Stride+self.Kernel, 
                                                       w*self.Stride:w*self.Stride+self.Kernel])
                        # find the max element index and assign to M
                        i,j = np.unravel_index(X[n, c, h*self.Stride:h*self.Stride+self.Kernel, 
                                                       w*self.Stride:w*self.Stride+self.Kernel].argmax(), 
                                               (self.Kernel,self.Kernel))
                        M[n, c, h*self.Stride+i, w*self.Stride+j] = 1
        self.cache = M
        return Y

    def _backward(self, up_grad):
        # M.shape = X.shape
        M = self.cache
        N, C_in, H, W = M.shape
        up_grad = np.array(up_grad)

        dX = np.zeros(M.shape)
        for n in range(N):
            for c in range(C_in):
                dX[n,c,:,:] = up_grad[n,c,:,:].repeat(self.Kernel, axis=0).repeat(self.Kernel, axis=1)
        dX = dX*M
        if self.ori_shape:
            dX = dX[:,:,:H-1, :W-1]
        return dX