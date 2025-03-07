import numpy as np
from module import module, XavierUniform, Zeros


class Linear(module):
    def __init__(self, d_in, d_out, w_init=XavierUniform(), b_init=Zeros()):

        super().__init__()

        self.initializers = {
            "weight": w_init,
            'bias': b_init,
        }

        self.input = None
        self.d_in = d_in
        self.d_out = d_out

        if d_in:
            self.shapes = {
                "weight": [d_in, d_out],
                "bias": [d_out]
            }

            self._init_params()

    def _forward(self, inputs):
        if not self.is_init:
            d_in = inputs.shape[-1]
            self.shapes = {
                "weight": [d_in, self.d_out],
                "bias": [self.d_out]
            }
            self.d_in = d_in
            self._init_params()

        # `@` is the matrix multiplication operator in NumPy
        out = inputs @ self.params['weight'] + self.params['bias']
        self.input = inputs
        return out

    def _backward(self, grad):
        ##############################################################################
        #                  TODO: You need to complete the code here                  #
        ##############################################################################
        # YOUR CODE HERE
        raise NotImplementedError()
        ##############################################################################
        #                              END OF YOUR CODE                              #
        ##############################################################################

    @property
    def param_names(self):
        return ('weight', 'bias')

    @property
    def weight(self):
        return self.params['weight']

    @property
    def bias(self):
        return self.params['bias']

if __name__ == '__main__':
    x = np.array(
        [0.41259363, -0.40173373, -0.9616683, 0.32021663, 0.30066854], dtype=np.float32)
    w = np.array([[-0.29742905, -0.4652604,  0.03716598],
                [0.63429886,  0.46831214,  0.22899507],
                [0.7614463,  0.45421863, -0.7652458],
                [0.6237591,  0.71807355,  0.81113386],
                [-0.34458044,  0.094055,  0.70938754]], dtype=np.float32)
    b = np.array([0., 0., 0.], dtype=np.float32)

    l = Linear(5, 3)
    l.params['weight'] = w
    l.params['bias'] = b
    l.is_init = True
    y = l._forward(x)
    l._backward(y)

    grad_b = np.array([-1.0136619, -0.5586895,  1.1322811], dtype=np.float32)
    grad_w = np.array([[-0.41823044, -0.23051172,  0.46717197],
                    [0.40722215,  0.2244444, -0.4548755],
                    [0.9748065,  0.53727394, -1.0888789],
                    [-0.32459137, -0.17890166,  0.36257523],
                    [-0.30477622, -0.16798034,  0.3404413]], dtype=np.float32)

    assert (np.abs(l.grads['bias'] - grad_b) < 1e-5).all()
    assert (np.abs(l.grads['weight'] - grad_w) < 1e-5).all()
    print('success!')