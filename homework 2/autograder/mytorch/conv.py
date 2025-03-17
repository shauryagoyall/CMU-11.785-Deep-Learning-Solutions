# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np


class Conv1D():
    def __init__(self, in_channel, out_channel, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_size)
        Return:
            out (np.array): (batch_size, out_channel, output_size)
        """
        batch_size, _, input_size = x.shape
        output_size = (input_size - self.kernel_size) // self.stride + 1
        out = np.zeros((batch_size, self.out_channel, output_size))

        for i in range(output_size):
            start = i * self.stride
            end = start + self.kernel_size
            out[:, :, i] = np.tensordot(x[:, :, start:end], self.W, axes=([1, 2], [1, 2])) + self.b

        self.x = x  # Store input for backward pass
        return out

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_size)
        Return:
            dx (np.array): (batch_size, in_channel, input_size)
        """
        batch_size, _, output_size = delta.shape
        dx = np.zeros_like(self.x)
        self.dW.fill(0)
        self.db.fill(0)

        for i in range(output_size):
            start = i * self.stride
            end = start + self.kernel_size

            self.dW += np.tensordot(delta[:, :, i], self.x[:, :, start:end], axes=([0], [0]))
            self.db += np.sum(delta[:, :, i], axis=0)

            dx[:, :, start:end] += np.tensordot(delta[:, :, i], self.W, axes=([1], [0]))

        return dx


class Conv2D():
    def __init__(self, in_channel, out_channel,
                 kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_width, input_height)
        Return:
            out (np.array): (batch_size, out_channel, output_width, output_height)
        """
        batch_size, _, input_width, input_height = x.shape
        output_width = (input_width - self.kernel_size) // self.stride + 1
        output_height = (input_height - self.kernel_size) // self.stride + 1
        out = np.zeros((batch_size, self.out_channel, output_width, output_height))

        for i in range(output_width):
            for j in range(output_height):
                start_w = i * self.stride
                end_w = start_w + self.kernel_size
                start_h = j * self.stride
                end_h = start_h + self.kernel_size

                out[:, :, i, j] = np.tensordot(
                    x[:, :, start_w:end_w, start_h:end_h], self.W, axes=([1, 2, 3], [1, 2, 3])
                ) + self.b

        self.x = x  # Store input for backward pass
        return out

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_width, output_height)
        Return:
            dx (np.array): (batch_size, in_channel, input_width, input_height)
        """
        batch_size, _, output_width, output_height = delta.shape
        dx = np.zeros_like(self.x)
        self.dW.fill(0)
        self.db.fill(0)

        for i in range(output_width):
            for j in range(output_height):
                start_w = i * self.stride
                end_w = start_w + self.kernel_size
                start_h = j * self.stride
                end_h = start_h + self.kernel_size

                self.dW += np.tensordot(delta[:, :, i, j], self.x[:, :, start_w:end_w, start_h:end_h], axes=([0], [0]))
                self.db += np.sum(delta[:, :, i, j], axis=0)

                dx[:, :, start_w:end_w, start_h:end_h] += np.tensordot(delta[:, :, i, j], self.W, axes=([1], [0]))

        return dx



# class Conv2D_dilation():
#     def __init__(self, in_channel, out_channel,
#                  kernel_size, stride, padding=0, dilation=1,
#                  weight_init_fn=None, bias_init_fn=None):
#         """
#         Much like Conv2D, but takes into account both padding and dilation.
#         The dilated kernel size is computed as:
#             kernel_dilated = (kernel_size - 1) * dilation + 1
#         and the dilated kernel is built by inserting zeros between the original weights.
#         """
#         self.in_channel = in_channel
#         self.out_channel = out_channel
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.dilation = dilation
#
#         # Compute the new (dilated) kernel size.
#         self.kernel_dilated = (kernel_size - 1) * dilation + 1
#
#         # Initialize weight and bias
#         if weight_init_fn is None:
#             self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size, kernel_size))
#         else:
#             self.W = weight_init_fn(out_channel, in_channel, kernel_size, kernel_size)
#
#         # Initialize the dilated version of W (will be built in forward)
#         self.W_dilated = np.zeros((self.out_channel, self.in_channel, self.kernel_dilated, self.kernel_dilated))
#
#         if bias_init_fn is None:
#             self.b = np.zeros(out_channel)
#         else:
#             self.b = bias_init_fn(out_channel)
#
#         # Gradients
#         self.dW = np.zeros(self.W.shape)
#         self.db = np.zeros(self.b.shape)
#
#     def __call__(self, x):
#         return self.forward(x)
#
#     def forward(self, x):
#         """
#         Forward pass of the dilated convolution operation.
#         Arguments:
#             x (np.array): (batch_size, in_channel, input_width, input_height)
#         Returns:
#             out (np.array): (batch_size, out_channel, output_width, output_height)
#         """
#         batch_size = x.shape[0]
#         in_channel = x.shape[1]
#         input_width = x.shape[-2]
#         input_height = x.shape[-1]
#
#         # Apply padding to the input
#         x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
#                           'constant', constant_values=0)
#         self.x_padded = x_padded
#
#         # Calculate the new padded width and height
#         input_width_padded = input_width + 2 * self.padding
#         input_height_padded = input_height + 2 * self.padding
#
#         # Apply dilation to the kernel
#         for i in range(self.out_channel):
#             for j in range(self.in_channel):
#                 kernel_idx_i = 0
#                 for m in range(0, self.kernel_dilated, self.dilation):
#                     kernel_idx_j = 0
#                     for n in range(0, self.kernel_dilated, self.dilation):
#                         self.W_dilated[i, j, m, n] = self.W[i, j, kernel_idx_i, kernel_idx_j]
#                         kernel_idx_j += 1
#                     kernel_idx_i += 1
#
#         # Compute output dimensions
#         output_width = (input_width_padded - self.kernel_dilated) // self.stride + 1
#         output_height = (input_height_padded - self.kernel_dilated) // self.stride + 1
#         output = np.zeros((batch_size, self.out_channel, output_width, output_height))
#
#         # Perform the actual convolution operation
#         for i in range(batch_size):
#             for j in range(self.out_channel):
#                 for m in range(output_width):
#                     for n in range(output_height):
#                         start_x = m * self.stride
#                         end_x = start_x + self.kernel_dilated
#                         start_y = n * self.stride
#                         end_y = start_y + self.kernel_dilated
#
#                         region = x_padded[i, :, start_x:end_x, start_y:end_y]
#                         output[i, j, m, n] = np.sum(self.W_dilated[j] * region) + self.b[j]
#
#         return output
#
#     def backward(self, delta):
#         """
#         Backward pass for calculating gradients.
#         Arguments:
#             delta (np.array): Gradient of the loss with respect to the output
#         Returns:
#             dx (np.array): Gradient of the loss with respect to the input
#         """
#         batch_size = delta.shape[0]
#         output_width = delta.shape[2]
#         output_height = delta.shape[3]
#         self.dx = np.zeros_like(self.x_padded)
#         self.dW_dilated = np.zeros_like(self.W_dilated)
#
#         # Compute gradient with respect to the bias
#         for i in range(batch_size):
#             for j in range(self.out_channel):
#                 for m in range(output_width):
#                     for n in range(output_height):
#                         self.db[j] += delta[i, j, m, n]
#
#         # Compute gradient with respect to weights and input
#         for i in range(batch_size):
#             for m in range(output_width):
#                 for n in range(output_height):
#                     for c in range(self.out_channel):
#                         for l in range(self.in_channel):
#                             for m_ in range(self.kernel_dilated):
#                                 for n_ in range(self.kernel_dilated):
#                                     start_x = m * self.stride
#                                     start_y = n * self.stride
#                                     region = self.x_padded[i, l, start_x + m_: start_x + m_ + 1,
#                                              start_y + n_: start_y + n_ + 1]
#                                     self.dW_dilated[c, l, m_, n_] += region * delta[i, c, m, n]
#                                     self.dx[i, l, start_x + m_, start_y + n_] += self.W_dilated[c, l, m_, n_] * delta[
#                                         i, c, m, n]
#
#         # Store the gradient of the weights in the original un-dilated shape
#         for i in range(self.out_channel):
#             for j in range(self.in_channel):
#                 kernel_idx_i = 0
#                 for m in range(0, self.dW_dilated.shape[-2], self.dilation):
#                     kernel_idx_j = 0
#                     for n in range(0, self.dW_dilated.shape[-1], self.dilation):
#                         self.dW[i, j, kernel_idx_i, kernel_idx_j] = self.dW_dilated[i, j, m, n]
#                         kernel_idx_j += 1
#                     kernel_idx_i += 1
#
#         # Remove the padding from the input gradient
#         self.dx = self.dx[:, :, self.padding:-self.padding, self.padding:-self.padding]
#
#         return self.dx


def im2col_dilated(x, kernel_size, stride, dilation):
    """
    Rearranges input patches into columns, taking dilation into account.
    x: (batch, channels, H, W)
    Returns:
        cols: (batch, channels*kernel_size*kernel_size, out_height*out_width)
        out_height, out_width: spatial dimensions of the output
    """
    batch, channels, H, W = x.shape
    kernel_extent = (kernel_size - 1) * dilation + 1
    out_height = (H - kernel_extent) // stride + 1
    out_width = (W - kernel_extent) // stride + 1

    # Create indices for the kernel window.
    i0 = np.repeat(np.arange(kernel_size), kernel_size) * dilation  # shape: (k*k,)
    j0 = np.tile(np.arange(kernel_size), kernel_size) * dilation       # shape: (k*k,)
    i1 = stride * np.arange(out_height)  # shape: (out_height,)
    j1 = stride * np.arange(out_width)   # shape: (out_width,)

    # Reshape indices so they can broadcast correctly.
    i = i0.reshape(-1, 1, 1) + i1.reshape(1, -1, 1)  # shape: (k*k, out_height, 1)
    j = j0.reshape(-1, 1, 1) + j1.reshape(1, 1, -1)  # shape: (k*k, 1, out_width)

    cols = x[:, :, i, j]  # Expected shape: (batch, channels, k*k, out_height, out_width)
    cols = cols.reshape(batch, channels * kernel_size * kernel_size, out_height * out_width)
    return cols, out_height, out_width


def col2im_dilated(cols, x_shape, kernel_size, stride, dilation, out_height, out_width):
    """
    Converts columnized patches back to the image shape.
    cols: (batch, channels*kernel_size*kernel_size, out_height*out_width)
    x_shape: shape of the padded input (batch, channels, H, W)
    Returns:
        dx: gradient tensor with shape x_shape
    """
    batch, channels, H, W = x_shape
    dx = np.zeros(x_shape)
    # Create the same indices as in im2col_dilated.
    i0 = np.repeat(np.arange(kernel_size), kernel_size) * dilation
    j0 = np.tile(np.arange(kernel_size), kernel_size) * dilation
    i1 = stride * np.arange(out_height)
    j1 = stride * np.arange(out_width)

    i = i0.reshape(-1, 1, 1) + i1.reshape(1, -1, 1)  # shape: (k*k, out_height, 1)
    j = j0.reshape(-1, 1, 1) + j1.reshape(1, 1, -1)  # shape: (k*k, 1, out_width)

    cols_reshaped = cols.reshape(batch, channels, kernel_size * kernel_size, out_height, out_width)
    # Accumulate the gradients for overlapping regions.
    for b in range(batch):
        for c in range(channels):
            np.add.at(dx[b, c], (i, j), cols_reshaped[b, c])
    return dx


class Conv2D_dilation():
    def __init__(self, in_channel, out_channel,
                 kernel_size, stride, padding=0, dilation=1,
                 weight_init_fn=None, bias_init_fn=None):
        """
        Dilated convolution layer using im2col/col2im for efficient computation.
        """
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        # Effective kernel size with dilation.
        self.kernel_dilated = (kernel_size - 1) * (dilation - 1) + kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Performs a forward pass of the dilated convolution.
        x: (batch, in_channel, height, width)
        Returns:
            output: (batch, out_channel, out_height, out_width)
        """
        # Pad the input.
        x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                          mode='constant', constant_values=0)
        self.x_padded = x_padded
        self.x_shape = x_padded.shape

        # Convert input into columns.
        cols, out_height, out_width = im2col_dilated(x_padded, self.kernel_size, self.stride, self.dilation)
        self.cols = cols
        self.out_height = out_height
        self.out_width = out_width

        # Reshape the weights.
        W_col = self.W.reshape(self.out_channel, -1)
        batch = x.shape[0]
        out = np.empty((batch, self.out_channel, cols.shape[-1]))
        # Convolve via matrix multiplication.
        for b in range(batch):
            out[b] = W_col @ cols[b] + self.b.reshape(-1, 1)
        out = out.reshape(batch, self.out_channel, out_height, out_width)
        return out

    def backward(self, delta):
        """
        Backward pass for dilated convolution.
        delta: (batch, out_channel, out_height, out_width) gradient w.r.t. output.
        Returns:
            dx: gradient w.r.t. the original input.
        """
        batch = delta.shape[0]
        delta_reshaped = delta.reshape(batch, self.out_channel, -1)
        W_col = self.W.reshape(self.out_channel, -1)

        # Compute gradient with respect to weights.
        dW = np.zeros_like(W_col)
        for b in range(batch):
            dW += delta_reshaped[b] @ self.cols[b].T
        self.dW = dW.reshape(self.W.shape)

        # Gradient with respect to bias.
        self.db = delta.sum(axis=(0, 2, 3))

        # Compute gradient with respect to input (columns).
        dcols = np.empty_like(self.cols)
        for b in range(batch):
            dcols[b] = W_col.T @ delta_reshaped[b]

        # Convert columns back to the original image shape.
        dx_padded = col2im_dilated(dcols, self.x_shape, self.kernel_size, self.stride,
                                   self.dilation, self.out_height, self.out_width)
        # Remove padding.
        if self.padding > 0:
            dx = dx_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dx = dx_padded
        return dx

class Flatten():
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, in_width)
        Return:
            out (np.array): (batch_size, in_channel * in_width)
        """
        self.b, self.c, self.w = x.shape
        # Flatten the input into a 2D array (batch_size, in_channel * in_width)
        out = x.reshape(self.b, self.c * self.w)
        return out

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, in_channel * in_width)
        Return:
            dx (np.array): (batch_size, in_channel, in_width)
        """
        # Reshape the delta back into the original 3D shape (batch_size, in_channel, in_width)
        dx = delta.reshape(self.b, self.c, self.w)
        return dx


