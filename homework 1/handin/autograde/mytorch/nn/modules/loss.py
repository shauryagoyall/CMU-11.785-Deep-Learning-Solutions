import numpy as np

class MSELoss:
    
    def forward(self, A, Y):
    
        self.A = A
        self.Y = Y
        N      = A.shape[0]
        C      = A.shape[1]
        se     = np.multiply ( self.A-self.Y, self.A-self.Y) # TODO
        # sse    = np.ones((1,N)) @ se @ np.ones((C,1)) # TODO ## not necessary?
        sse = np.sum(se)
        mse    = sse/(N*C)
        
        return mse
    
    def backward(self):
    
        dLdA = self.A - self.Y
        
        return dLdA

class CrossEntropyLoss:
    
    def forward(self, A, Y):
    
        self.A   = A
        self.Y   = Y
        N        = A.shape[0]
        C        = A.shape[1]
        # Ones_C   = np.ones((C, 1), dtype="f") ## not needed as just initialize a C x C matrix and for summing do directly
        # Ones_N   = np.ones((N, 1), dtype="f")

        self.softmax     = np.divide(np.exp(A), np.dot(np.exp(A), np.ones((C,C), dtype="f"))) # TODO
        crossentropy     = np.multiply(-self.Y , np.log(self.softmax)) # TODO
        sum_crossentropy = np.sum(crossentropy) # TODO
        L = sum_crossentropy / N
        
        return L
    
    def backward(self):
    
        dLdA = self.softmax - self.Y # TODO
        
        return dLdA
