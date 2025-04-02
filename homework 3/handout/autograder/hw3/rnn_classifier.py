import numpy as np
import sys

sys.path.append("mytorch")
from rnn_cell import *
from linear import *


class RNNPhonemeClassifier(object):
    """RNN Phoneme Classifier class."""

    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # TODO: Understand then uncomment this code :)
        self.rnn = [
            RNNCell(input_size, hidden_size) if i == 0
                else RNNCell(hidden_size, hidden_size)
                    for i in range(num_layers)
        ]
        self.output_layer = Linear(hidden_size, output_size)

        # store hidden states at each time step, [(seq_len+1) * (num_layers, batch_size, hidden_size)]
        self.hiddens = []

    def init_weights(self, rnn_weights, linear_weights):
        """Initialize weights.

        Parameters
        ----------
        rnn_weights:
                    [
                        [W_ih_l0, W_hh_l0, b_ih_l0, b_hh_l0],
                        [W_ih_l1, W_hh_l1, b_ih_l1, b_hh_l1],
                        ...
                    ]

        linear_weights:
                        [W, b]

        """
        for i, rnn_cell in enumerate(self.rnn):
            rnn_cell.init_weights(*rnn_weights[i])
        self.output_layer.W = linear_weights[0]
        self.output_layer.b = linear_weights[1].reshape(-1, 1)

    def __call__(self, x, h_0=None):
        return self.forward(x, h_0)

    def forward(self, x, h_0=None):
        """RNN forward, multiple layers, multiple time steps.

        Parameters
        ----------
        x: (batch_size, seq_len, input_size)
            Input

        h_0: (num_layers, batch_size, hidden_size)
            Initial hidden states. Defaults to zeros if not specified

        Returns
        -------
        logits: (batch_size, output_size) 

        Output (y): logits

        """
        # Get the batch size and sequence length, and initialize the hidden
        # vectors given the paramters.
        batch_size, seq_len = x.shape[0], x.shape[1]
        if h_0 is None:
            hidden = np.zeros((self.num_layers, batch_size, self.hidden_size), dtype=float)
        else:
            hidden = h_0

        # Save x and append the hidden vector to the hiddens list
        self.x = x
        self.hiddens.append(hidden.copy())
        logits = None

        ### Add your code here --->
        # (More specific pseudocode may exist in lecture slides)
        # Iterate through the sequence
        #   Iterate over the length of your self.rnn (through the layers)
        #       Run the rnn cell with the correct parameters and update
        #       the parameters as needed. Update hidden.
        #   Similar to above, append a copy of the current hidden array to the hiddens list

        ### Forward Pass ###
        for t in range(seq_len):  # Iterate over time steps
            x_t = x[:, t, :]  # Get input for time step t (shape: batch_size x input_size)

            for layer in range(self.num_layers):  # Iterate over RNN layers
                h_prev_t = hidden[layer]  # Previous hidden state for this layer
                h_t = self.rnn[layer](x_t, h_prev_t)  # Forward pass through RNNCell
                hidden[layer] = h_t  # Update hidden state

                # Input to next layer is the hidden state from the current layer
                x_t = h_t

                # Store the updated hidden states for this time step
            self.hiddens.append(hidden.copy())

        # Get the outputs from the last time step using the linear layer
        logits = self.output_layer(hidden[-1])  # Use last layer's hidden state

        return logits

    def backward(self, delta):
        """RNN Back Propagation Through Time (BPTT).

        Parameters
        ----------
        delta: (batch_size, hidden_size)

        gradient: dY(seq_len-1)
                gradient w.r.t. the last time step output.

        Returns
        -------
        dh_0: (num_layers, batch_size, hidden_size)

        gradient w.r.t. the initial hidden states

        """
        # Initilizations
        batch_size, seq_len = self.x.shape[0], self.x.shape[1]
        dh = np.zeros((self.num_layers, batch_size, self.hidden_size), dtype=float)
        dh[-1] = self.output_layer.backward(delta)

        # BPTT
        for t in reversed(range(seq_len)):
            dx = None

            for layer in reversed(range(self.num_layers)):
                h_t = self.hiddens[t + 1][layer]  # Current hidden state at time t
                h_prev_t = self.hiddens[t][layer]  # Previous hidden state at time t-1

                h_prev_l = self.x[:, t, :] if layer == 0 else self.hiddens[t + 1][layer - 1] # Previous later hidden state at time t

                dx_t, dh_prev_t = self.rnn[layer].backward(dh[layer], h_t, h_prev_l, h_prev_t)

                dh[layer] = dh_prev_t
                if layer > 0:
                    dh[layer - 1] += dx_t

        return dh / batch_size
