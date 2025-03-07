import numpy as np
from module import module, XavierUniform, Zeros

class Conv2D(module):
    """
    Implement 2D convolution layer
    :param kernel: A list/tuple of int that has length 4 (in_channels, height, width, out_channels)
    :param stride: A list/tuple of int that has length 2 (height, width)
    :param padding: String ["SAME", "VALID"]
    :param w_init: weight initializer
    :param b_init: bias initializer
    """

    def __init__(self,
                 kernel,
                 stride=(1, 1),
                 padding="SAME",
                 w_init=XavierUniform(),
                 b_init=Zeros()):
        super().__init__()

        self.kernel_shape = kernel
        self.stride = stride
        self.initializers = {"weight": w_init, "bias": b_init}
        self.shapes = {"weight": self.kernel_shape,
                       "bias": self.kernel_shape[-1]}

        self.padding_mode = padding
        assert padding in ['SAME', 'VALID']
        if padding == 'SAME' and stride != (1, 1):
            raise RuntimeError(
                "padding='SAME' is not supported for strided convolutions.")
        self.padding = None

        self._init_params()

    def _forward(self, inputs):
        """
        :param inputs:  shape (batch_size, in_c, in_h, in_w)
        :return outputs: shape (batch_size, out_c, out_h, out_w)
        where batch size is the number of images
        """
        assert len(
            inputs.shape) == 4, 'Expected shape of inputs is (batch_size, in_c, in_h, in_w).'
        in_c, k_h, k_w, out_c = self.kernel_shape
        s_h, s_w = self.stride
        X = self._inputs_preprocess(inputs)
        bsz, _, h, w = X.shape

        out_h = (h - k_h) // s_h + 1
        out_w = (w - k_w) // s_w + 1
        Y = np.zeros([bsz, out_c, out_h, out_w])
        for in_c_i in range(in_c):
            for out_c_i in range(out_c):
                kernel = self.params['weight'][in_c_i, :, :, out_c_i]
                for r in range(out_h):
                    r_start = r * s_h
                    for c in range(out_w):
                        c_start = c * s_w
                        patch = X[:, in_c_i, r_start: r_start +
                                  k_h, c_start: c_start+k_w] * kernel
                        Y[:, out_c_i, r,
                            c] += patch.reshape(bsz, -1).sum(axis=-1)
        self.input = inputs
        return Y + self.params['bias'].reshape(1, -1, 1, 1)

    def _backward(self, grad):
        """
        Compute gradients w.r.t layer parameters and backward gradients.
        :param grad: gradients from previous layer 
            with shape (batch_size, out_c, out_h, out_w)
        :return d_in: gradients to next layers 
            with shape (batch_size, in_c, in_h, in_w)
        """
        assert len(
            grad.shape) == 4, 'Expected shape of upstream gradient is (batch_size, out_c, out_h, out_w)'
        ##############################################################################
        #                  TODO: You need to complete the code here                  #
        ##############################################################################
        # YOUR CODE HERE
        raise NotImplementedError()
        ##############################################################################
        #                              END OF YOUR CODE                              #
        ##############################################################################

    def _inputs_preprocess(self, inputs):
        _, _, in_h, in_w = inputs.shape
        _, k_h, k_w, _ = self.kernel_shape
        # padding calculation
        if self.padding is None:
            self.padding = self.get_padding_2d(
                (in_h, in_w), (k_h, k_w), self.stride, self.padding_mode)
        return np.pad(inputs, pad_width=self.padding, mode="constant")

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

    @property
    def param_names(self):
        return "weight", "bias"

    @property
    def weight(self):
        return self.params['weight']

    @property
    def bias(self):
        return self.params['bias']
    

if __name__ == '__main__':
    x = np.array([[[[-1.75957111,  0.0085911,  0.30235818],
                    [-1.05931037,  0.75555462, -2.03922536],
                    [0.86653209, -0.56438439, -1.68797524]],

                [[-0.74832044,  0.21611616,  0.571611],
                    [-1.61335018, -0.37620906,  1.0353189],
                    [-0.26074537,  1.98065489, -1.30691981]]],


                [[[0.32680334,  0.29817393,  2.25433969],
                    [-0.16831957, -0.98864486,  0.36653],
                    [1.52712821,  1.19630751, -0.02024759]],

                [[0.48080474, -1.15229596, -0.95228854],
                    [-1.68168285, -2.86668484, -0.34833734],
                    [0.73179971,  1.69618114,  1.33524773]]]], dtype=np.float32)
    w = np.array([[[[-0.322831,  0.38674766,  0.32847992,  0.3846352],
                    [-0.21158722, -0.53467643, -0.28443742, -0.20367976]],

                [[0.4973593, -0.30178958, -0.02311361, -0.53795236],
                    [-0.1229187, -0.12866518, -0.40432686,  0.5104686]]],


                [[[0.19288206, -0.49516755, -0.26484585, -0.35625377],
                    [0.5058061, -0.17490079, -0.40337119,  0.10058666]],

                [[-0.24815331,  0.34114942, -0.06982624,  0.4017606],
                    [0.16874631, -0.42147416,  0.43324274,  0.16369782]]]], dtype=np.float32)
    b = np.array([0., 0., 0., 0.], dtype=np.float32)

    l = Conv2D(kernel=(2, 2, 2, 4), padding='SAME', stride=(1, 1))
    l.params['weight'] = w
    l.params['bias'] = b
    l.is_init = True
    y = l._forward(x)
    l._backward(y)

    grad_b = np.array([-0.49104962,  1.4335476,  2.70048173, -
                    0.0098734], dtype=np.float32)
    grad_w = np.array([[[[-3.0586028,   7.7819834,   1.3951588,   5.9249396],
                        [-1.5760803, -10.541515,  -2.694372,  -3.9848034]],

                        [[2.9096646,   0.6696263,   8.230143,  -0.3434674],
                        [-2.9487448,  -3.264796,  -1.1822633,   4.1672387]]],


                    [[[3.7202294,  -5.4176836, -10.34358,  -6.4479938],
                        [7.0336857,  -0.41946477,  -8.181945,   3.0968976]],

                        [[0.25020388,  13.39637,   5.8576417,  12.522377],
                        [3.360495,  -6.597466,   8.375789,   3.8179488]]]], dtype=np.float32)

    assert (np.abs(l.grads['bias'] - grad_b) < 1e-5).all()
    assert (np.abs(l.grads['weight'] - grad_w) < 1e-5).all()
    print('success!')