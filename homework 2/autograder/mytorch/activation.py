import numpy as np


class Identity:
    
    def forward(self, Z):
    
        self.A = Z
        
        return self.A
    
    def backward(self):
    
        dAdZ = np.ones(self.A.shape, dtype="f")
        
        return dAdZ


class Sigmoid:
    def __call__(self, x):
        # Do not modify this method
        return self.forward(x)
    def forward(self, Z):
    
        self.A = 1/(1 + np.exp(-Z)) # TODO
        
        return self.A
    
    def backward(self):
    
        dAdZ = self.A * (1 - self.A) # TODO
        
        return dAdZ


class Tanh:

    def __call__(self, x):
        # Do not modify this method
        return self.forward(x)

    def forward(self, Z):
    
        self.A = np.tanh(Z) # TODO
        
        return self.A
    
    def backward(self):
    
        dAdZ = 1 - (self.A)**2 # TODO
        
        return dAdZ


class ReLU:

    def __call__(self, x):
        # Do not modify this method
        return self.forward(x)

    def forward(self, Z):
    
        self.A = np.maximum(Z, np.zeros_like(Z, dtype=float)) # TODO
        
        return self.A
    
    def backward(self):
    
        dAdZ = np.where(self.A != 0, 1, 0) # TODO
        
        return dAdZ
        
        
