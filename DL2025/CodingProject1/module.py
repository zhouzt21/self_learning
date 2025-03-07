################### DO NOT MODIFIED THE CODE ###################
import numpy as np


class Initializer:

    def __call__(self, shape):
        return self.init(shape).astype(np.float32)

    def init(self, shape):
        raise NotImplementedError


class XavierUniform(Initializer):
    """
    Implement the Xavier method described in
    "Understanding the difficulty of training deep feedforward neural networks"
    Glorot, X. & Bengio, Y. (2010)
    Weights will have values sampled from uniform distribution U(-a, a) where
    a = gain * sqrt(6.0 / (num_in + num_out))
    """

    def __init__(self, gain=1.0):
        self._gain = gain

    def init(self, shape):
        fan_in, fan_out = self.get_fans(shape)
        a = self._gain * np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(low=-a, high=a, size=shape)

    def get_fans(self, shape):
        fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
        fan_out = shape[1] if len(shape) == 2 else shape[0]
        return fan_in, fan_out


class Constant(Initializer):

    def __init__(self, val):
        self._val = val

    def init(self, shape):
        return np.full(shape=shape, fill_value=self._val)


class Zeros(Constant):

    def __init__(self):
        super(Zeros, self).__init__(0.0)


class module():

    def __init__(self, **kwargs):
        self.params = {p: None for p in self.param_names}
        self.ut_params = {p: None for p in self.ut_param_names}

        self.grads = {}
        self.shapes = {}

        self.training = True
        self.is_init = False

    def _forward(self, X, **kwargs):
        raise NotImplementedError

    def _backward(self, d, **kwargs):
        raise NotImplementedError

    def set_phase(self, phase):
        self.training = phase.lower() == "train"

    @property
    def name(self):
        return self.__class__.__name__

    def __repr__(self):
        shape = None if not self.shapes else self.shapes
        return "module: %s \t shape: %s" % (self.name, shape)

    @property
    def param_names(self):
        return ()

    def _init_params(self):
        for p in self.param_names:
            self.params[p] = self.initializers[p](self.shapes[p])
        self.is_init = True

    @property
    def ut_param_names(self):
        return ()


class Activation(module):

    def __init__(self):
        super().__init__()
        self.inputs = None

    def _forward(self, inputs):
        self.inputs = inputs
        return self.func(inputs)

    def func(self, x):
        raise NotImplementedError


def hello():
    print("Hello from your TAs!")
