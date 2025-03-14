# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np

class Dropout(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x, train=True):

        if train:
            # TODO: Generate mask and apply to x
            self.mask = np.random.binomial(1, 1 - self.p, size=x.shape) ## taking 1-p as p is prob of dropping ie. 0 so success ie 1 is 1-p
            x = x * self.mask / (1 - self.p)
            return x
        else:
            return x
		
    def backward(self, delta):
        # TODO: Multiply mask with delta and return
        delta = delta * self.mask
        return delta
