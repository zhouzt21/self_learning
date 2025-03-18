
import numpy as np
from module import module, XavierUniform, Zeros

class MaxPool2D(module):

    def __init__(self, pool_size, stride, padding="VALID"):
        """
        Implement 2D max-pooling layer
        :param pool_size: A list/tuple of 2 integers (pool_height, pool_width)
        :param stride: A list/tuple of 2 integers (stride_height, stride_width)
        :param padding: A string ("SAME", "VALID")
        """
        super().__init__()
        self.kernel_shape = pool_size
        self.stride = stride

        self.padding_mode = padding
        self.padding = None

    def _forward(self, inputs):
        """
        :param inputs:  shape (batch_size, in_c, in_h, in_w)
        """
        s_h, s_w = self.stride
        k_h, k_w = self.kernel_shape
        batch_sz, in_c, in_h, in_w = inputs.shape

        # zero-padding
        if self.padding is None:
            self.padding = self.get_padding_2d(
                (in_h, in_w), (k_h, k_w), self.stride, self.padding_mode)
        X = np.pad(inputs, pad_width=self.padding, mode="constant")
        padded_h, padded_w = X.shape[2:4]

        out_h = (padded_h - k_h) // s_h + 1
        out_w = (padded_w - k_w) // s_w + 1

        # construct output matrix and argmax matrix
        max_pool = np.empty(shape=(batch_sz, in_c, out_h, out_w))
        argmax = np.empty(shape=(batch_sz, in_c, out_h, out_w), dtype=int)
        for r in range(out_h):
            r_start = r * s_h
            for c in range(out_w):
                c_start = c * s_w
                pool = X[:, :, r_start: r_start+k_h, c_start: c_start+k_w]
                #  (batch_sz, in_c, k_h, k_w) -> (batch_sz, in_c, k_h*k_w)
                pool = pool.reshape((batch_sz, in_c, -1))

                # (batch_sz, in_c) -> (batch_sz, in_c, 1) index of max elements
                _argmax = np.argmax(pool, axis=2)[:, :, np.newaxis]
                argmax[:, :, r, c] = _argmax.squeeze(axis=2)

                # get max elements
                _max_pool = np.take_along_axis(
                    pool, _argmax, axis=2).squeeze(axis=2)
                max_pool[:, :, r, c] = _max_pool

        self.X_shape = X.shape
        self.out_shape = (out_h, out_w)
        self.argmax = argmax
        return max_pool

    def _backward(self, grad):
        ##############################################################################
        #                  TODO: You need to complete the code here                  #
        ##############################################################################
        dX = np.zeros(self.X_shape)
        s_h, s_w = self.stride
        k_h, k_w = self.kernel_shape
        out_h, out_w = self.out_shape

        for r in range(out_h):
            r_start = r * s_h
            for c in range(out_w):
                c_start = c * s_w
                # get the argmax of this window, size of (batch_size, in_c)
                window_argmax = self.argmax[:, :, r, c]
                # shape (batch_size, in_c)
                grad_curr = grad[:, :, r, c]
                # trans to 2D index
                row_offset = window_argmax // k_w
                col_offset = window_argmax % k_w

                for i in range(dX.shape[0]):      # batch size
                    for j in range(dX.shape[1]):  # channel
                        pos_r = r_start + row_offset[i, j]
                        pos_c = c_start + col_offset[i, j]
                        dX[i, j, pos_r, pos_c] += grad_curr[i, j]

        # unpadding
        pad_top, pad_bottom = self.padding[2]
        pad_left, pad_right = self.padding[3]
        d_input = dX[:, :, pad_top: dX.shape[2]-pad_bottom, pad_left: dX.shape[3]-pad_right]
        return d_input

        ##############################################################################
        #                              END OF YOUR CODE                              #
        ##############################################################################

    def get_padding_2d(self, in_shape, k_shape, stride, mode):

        def get_padding_1d(w, k, s):
            if mode == "SAME":
                pads = s * (w - 1) + k - w
                half = pads // 2
                padding = (half, half) if pads % 2 == 0 else (half, half + 1)
            else:
                padding = (0, 0)
            return padding

        h_pad = get_padding_1d(in_shape[0], k_shape[0], stride[0])
        w_pad = get_padding_1d(in_shape[1], k_shape[1], stride[1])
        return (0, 0), (0, 0), h_pad, w_pad
    
if __name__ == '__main__':
    x = np.array([[[[-2.23317337,  0.9750834, -1.30762567, -0.71442179],
                [0.24624013, -1.77593893, -0.43530428,  1.03446008],
                [1.58317228, -0.66459249,  0.54894879, -1.19706709],
                [0.06013156,  1.05886458,  0.26634763,  1.03497421]]],


              [[[2.20214308, -0.53358514,  0.96765812, -1.74976553],
                [-0.07049627,  0.88147726,  2.15051543, -0.78627764],
                  [1.19180886,  0.00468398, -1.74774108,  0.18564536],
                  [1.39397303, -1.0462731,  0.4786774, -0.51543751]]]], dtype=np.float32)

    l = MaxPool2D(pool_size=(2, 2), stride=(2, 2))
    l.is_init = True
    y = l._forward(x)
    grad_ = l._backward(y)

    grad = np.array([[[[0., 0.9750834, 0., 0.],
                    [0., 0., 0., 1.0344601],
                    [1.5831723, 0., 0., 0.],
                    [0., 0., 0., 1.0349742]]],


                    [[[2.2021432, 0., 0., 0.],
                    [0., 0., 2.1505153, 0.],
                        [0., 0., 0., 0.],
                        [1.393973, 0., 0.4786774, 0.]]]], dtype=np.float32)

    assert (np.abs(grad - grad_) < 1e-5).all()
    print('success!')