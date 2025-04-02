import numpy as np
from activation import *


class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, in_dim, hidden_dim):
        self.d = in_dim
        self.h = hidden_dim
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.brx = np.random.randn(h)
        self.bzx = np.random.randn(h)
        self.bnx = np.random.randn(h)

        self.brh = np.random.randn(h)
        self.bzh = np.random.randn(h)
        self.bnh = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbrx = np.zeros((h))
        self.dbzx = np.zeros((h))
        self.dbnx = np.zeros((h))

        self.dbrh = np.zeros((h))
        self.dbzh = np.zeros((h))
        self.dbnh = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx
        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def __call__(self, x, h_prev_t):
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h_prev_t: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        self.x = x
        self.hidden = h_prev_t
        

        
        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)


        # reset gate (r)
        self.r = self.r_act(self.Wrx @ x + self.brx + self.Wrh @ h_prev_t + self.brh)

        # update gate (z)
        self.z = self.z_act(self.Wzx @ x + self.bzx + self.Wzh @ h_prev_t + self.bzh)

        # new gate (n)
        self.n = self.h_act(self.Wnx @ x + self.bnx + self.r * (self.Wnh @ h_prev_t + self.bnh))

        # final hidden state (h_t)
        h_t = (1 - self.z) * self.n + self.z * h_prev_t

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.n.shape == (self.h,)
        assert h_t.shape == (self.h,)

        return h_t

    def backward(self, delta):
        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (1, input_dim)
            derivative of the loss wrt the input x.

        dh_prev_t: (1, hidden_dim)
            derivative of the loss wrt the input hidden h.

        """
        # 1) Reshape self.x and self.hidden to (input_dim, 1) and (hidden_dim, 1) respectively
        #    when computing self.dWs...
        # 2) Transpose all calculated dWs...
        # 3) Compute all of the derivatives
        # 4) Know that the autograder grades the gradients in a certain order, and the
        #    local autograder will tell you which gradient you are currently failing.

        # ADDITIONAL TIP:
        # Make sure the shapes of the calculated dWs and dbs  match the
        # initalized shapes accordingly




        dz = delta * (self.hidden - self.n) * self.z_act.backward()

        dn = delta * (1 - self.z)  * self.h_act.backward(self.n)

        dr = dn * (self.Wnh @ self.hidden + self.bnh) * self.r_act.backward()

        dht_1 = self.z * delta

        dz = dz.reshape(self.h, 1)  # Ensure shape is (hidden_dim, 1)
        dn = dn.reshape(self.h, 1)  # Ensure shape is (hidden_dim, 1)
        dr = dr.reshape(self.h, 1)  # Ensure shape is (hidden_dim, 1)


        dx = (self.Wrx.T @ dr + self.Wzx.T @ dz + self.Wnx.T @ dn).T
        dh_prev_t = (self.Wrh.T @ dr + self.Wzh.T @ dz + self.Wnh.T @ (self.r.reshape(-1, 1) * dn)).T
        dh_prev_t += dht_1

        self.x = self.x.reshape(self.d, 1)  # Reshape input to (input_dim, 1)
        self.hidden = self.hidden.reshape(self.h, 1)  # Reshape previous hidden state to (hidden_dim, 1)

        self.dWrx += dr @ self.x.T
        self.dWzx += dz @ self.x.T
        self.dWnx += dn @ self.x.T

        self.dWrh += dr @ self.hidden.T
        self.dWzh += dz @ self.hidden.T
        self.dWnh += (self.r.reshape(-1, 1) * dn) @ self.hidden.T

        self.dbrx += dr.flatten()
        self.dbzx += dz.flatten()
        self.dbnx += dn.flatten()

        self.dbrh += dr.flatten()
        self.dbzh += dz.flatten()
        self.dbnh += (self.r.reshape(-1, 1) * dn).flatten()

        assert dx.shape == (1, self.d)
        assert dh_prev_t.shape == (1, self.h)

        return dx, dh_prev_t
