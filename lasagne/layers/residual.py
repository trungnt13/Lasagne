import numpy as np
import theano
from theano import tensor

from .. import init
from .. import nonlinearities

from .base import Layer

from ..random import get_rng
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

__all__ = [
    "ResidualDenseLayer",
    "ResidualConv2DLayer",
]

class ResidualDenseLayer(Layer):

    """
    lasagne.layers.ResidualDenseLayer(incoming, num_units,
        W=init.GlorotUniform(), b=init.Constant(0.),
        noise=0., dropout=0., rescale=True,
        nonlinearity=nonlinearities.rectify, **kwargs)

    A fully connected Residual layer.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape

    num_units : int
        The number of units of the layer

    W : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the weights.
        These should be a matrix with shape ``(num_inputs, num_units)``.
        See :func:`lasagne.utils.create_param` for more information.

    b : Theano shared variable, expression, numpy array, callable or ``None``
        Initial value, expression or initializer for the biases. If set to
        ``None``, the layer will have no biases. Otherwise, biases should be
        a 1D array with shape ``(num_units,)``.
        See :func:`lasagne.utils.create_param` for more information.

    noise : float or tensor scalar
            (GaussianNoise) Standard deviation of added Gaussian noise

    dropout : float or scalar tensor
        (Dropout) The probability of setting a value to zero

    rescale : bool
        (Dropout) If true the input is rescaled with input / (1-p) when
        deterministic is False.

    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear.

    Examples
    --------
    >>> from lasagne.layers import InputLayer, ResidualDenseLayer
    >>> l_in = InputLayer((100, 20))
    >>> l1 = ResidualDenseLayer(l_in, num_units=[50, 20,40])

    Reference
    ---------
    Deep Residual Learning for Image Recognition (http://arxiv.org/abs/1512.03385)

    Notes
    -----
    Output shape from this layer is always the same as input shape.
    If the input to this layer has more than two axes, it will flatten the
    trailing axes.
    """

    def __init__(self, incoming, num_units, W=init.GlorotUniform(),
                 b=init.Constant(0.), noise=0., dropout=0., rescale=True,
                 nonlinearity=nonlinearities.rectify,
                 **kwargs):
        super(ResidualDenseLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)

        if not hasattr(num_units, '__len__'):
            num_units = [num_units]
        self.num_units = num_units

        output_dim = int(np.prod(self.input_shape[1:])) # same as input_dim

        # set Noise and dropout
        if noise is not None or dropout is not None:
            self._srng = RandomStreams(get_rng().randint(1, 2147462579))

        if 'SharedVariable' not in str(type(noise)):
            self.sigma = theano.shared(np.cast[theano.config.floatX](noise),
                                       name='sigma')
        self.sigma = noise

        if 'SharedVariable' not in str(type(dropout)):
            self.p = theano.shared(np.cast[theano.config.floatX](dropout),
                                   name='p')
        self.p = dropout
        self.rescale = rescale

        # create multiple layer intermediate layer
        self.W = []
        self.b = []
        last_dim = output_dim
        for dim in num_units:
            self.W.append(self.add_param(W, (last_dim, dim), name="W"))
            if b is None:
                self.b.append(None)
            else:
                self.b.append(self.add_param(b, (dim,), name="b",
                                    regularizable=False))
            last_dim = dim

        # create output layer
        self.W.append(self.add_param(W, (last_dim, output_dim), name="W"))
        if b is None:
            self.b.append(None)
        else:
            self.b.append(self.add_param(b, (output_dim,), name="b",
                                regularizable=False))

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, deterministic=False, **kwargs):
        activation = input
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            activation = input.flatten(2)

        # inner layers
        for W, b in zip(self.W[:-1], self.b[:-1]):
            activation = tensor.dot(activation, W)
            if b is not None:
                activation = activation + b.dimshuffle('x', 0)
            activation = self.nonlinearity(activation)
            # Gaussian noise
            if self.sigma is not None and not deterministic:
                print('noise')
                activation = activation + self._srng.normal(
                    activation.shape, avg=0.0, std=self.sigma)
            # Dropout
            if self.p is not None and not deterministic:
                print('dropout')
                retain_prob = 1 - self.p
                if self.rescale:
                    activation /= retain_prob

                # use nonsymbolic shape for dropout mask if possible
                activation = activation * self._srng.binomial(
                    activation.shape, p=retain_prob,
                    dtype=theano.config.floatX)

        # residual layer
        activation = tensor.dot(activation, self.W[-1])
        if b is not None:
            activation = activation + self.b[-1].dimshuffle('x', 0)
        return self.nonlinearity(activation).reshape(input.shape) + input

class ResidualConv2DLayer(Layer):

    """
    lasagne.layers.DenseLayer(incoming, num_units,
    W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
    nonlinearity=lasagne.nonlinearities.rectify, **kwargs)

    A fully connected layer.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape

    num_units : int
        The number of units of the layer

    W : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the weights.
        These should be a matrix with shape ``(num_inputs, num_units)``.
        See :func:`lasagne.utils.create_param` for more information.

    b : Theano shared variable, expression, numpy array, callable or ``None``
        Initial value, expression or initializer for the biases. If set to
        ``None``, the layer will have no biases. Otherwise, biases should be
        a 1D array with shape ``(num_units,)``.
        See :func:`lasagne.utils.create_param` for more information.

    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear.

    Examples
    --------
    >>> from lasagne.layers import InputLayer, DenseLayer
    >>> l_in = InputLayer((100, 20))
    >>> l1 = DenseLayer(l_in, num_units=50)

    Notes
    -----
    If the input to this layer has more than two axes, it will flatten the
    trailing axes. This is useful for when a dense layer follows a
    convolutional layer, for example. It is not necessary to insert a
    :class:`FlattenLayer` in this case.
    """

    def __init__(self, incoming, num_units, W=init.GlorotUniform(),
                 b=init.Constant(0.), nonlinearity=nonlinearities.rectify,
                 **kwargs):
        super(ResidualConv2DLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)

        if hasattr(num_units, '__len__'):
            num_units = [num_units]
        self.num_units = num_units

        self.output_dim = int(np.prod(self.input_shape[1:]))

        self.W = []
        self.b = []
        for u in self.num_units: # create multiple layer
            self.W.append(self.add_param(W, (self.output_dim, num_units), name="W"))
            if b is None:
                self.b.append(None)
            else:
                self.b.append(self.add_param(b, (num_units,), name="b",
                                        regularizable=False))
        self.W.append(self.add_param(W, (u, self.output_dim), name="W"))
        if b is None:
            self.b.append(None)
        else:
            self.b.append(self.add_param(b, (self.output_dim,), name="b",
                                        regularizable=False))

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_output_for(self, input, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)

        i = input
        for w, b in zip(self.W[:-1], self.b[:-1]):
            activation = tensor.dot(i, w)
            if b is not None:
                activation = activation + b.dimshuffle('x', 0)
            i = self.nonlinearity(activation)
        # final output
        activation = tensor.dot(i, self.W[-1])
        if self.b[-1] is not None:
            activation = activation + self.b[-1].dimshuffle('x', 0)
        return self.nonlinearity(activation) + i
