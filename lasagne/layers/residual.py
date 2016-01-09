import numpy as np
import theano
from theano import tensor

from .. import init
from .. import nonlinearities

from .dense import DenseLayer
from .noise import DropoutLayer, GaussianNoiseLayer
from .normalization import batch_norm
from .shape import ReshapeLayer, PadLayer
from .merge import ElemwiseSumLayer
from .special import NonlinearityLayer

__all__ = [
    "residual_dense",
    "residual_conv2d"
]

def residual_dense(incoming, num_units,
            W=init.GlorotUniform(), b=init.Constant(0.),
            nonlinearity=nonlinearities.rectify,
            noise=0., dropout=0., rescale=True,
            batch_normalization=True, **kwargs):
    '''
    Create set of fully connected Residual layers.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape

    num_units : int, list(int)
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

    noise : float or tensor scalar
            (GaussianNoise) Standard deviation of added Gaussian noise

    dropout : float or scalar tensor
        (Dropout) The probability of setting a value to zero

    rescale : bool
        (Dropout) If true the input is rescaled with input / (1-p) when
        deterministic is False.

    batch_norm : bool
        Apply batch normalization for each layer

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
    '''
    if not hasattr(num_units, '__len__'):
        num_units = [num_units]

    # create intermediate layer
    l = incoming
    for i in num_units:
        l = DenseLayer(l, num_units=i, W=W, b=b, nonlinearity=nonlinearity, **kwargs)
        if batch_normalization:
            l = batch_norm(l)
        if noise is not None:
            l = GaussianNoiseLayer(l, sigma=noise)
        if dropout is not None:
            l = DropoutLayer(l, p=dropout, rescale=rescale)

    # create output layer
    l = DenseLayer(l, num_units=int(np.prod(incoming.output_shape[1:])),
        W=W, b=b, nonlinearity=nonlinearity, **kwargs)
    l = ReshapeLayer(l, shape=([0],) + incoming.output_shape[1:])
    l = ElemwiseSumLayer((l, incoming))
    if batch_normalization:
        l = batch_norm(l)
    return l


def residual_block(layer, num_filters, filter_size=3, stride=1, num_layers=2):
    try:
        from .dnn import Conv2DDNNLayer as Conv2DLayer
    except:
        from .conv import Conv2DLayer

    conv = layer
    if (num_filters != layer.output_shape[1]) or (stride != 1):
        layer = Conv2DLayer(layer, num_filters, filter_size=1, stride=stride, pad=0, nonlinearity=None, b=None)
    for _ in range(num_layers):
        conv = Conv2DLayer(conv, num_filters, filter_size, stride=stride, pad='same')
        stride = 1
    nonlinearity = conv.nonlinearity
    conv.nonlinearity = nonlinearities.identity
    return NonlinearityLayer(ElemwiseSumLayer([conv, layer]), nonlinearity)

def residual_conv2d(l, increase_dim=False, projection=False):
    try:
        from .dnn import Conv2DDNNLayer as Conv2DLayer
        from .dnn import Pool2DDNNLayer as Pool2DLayer
    except:
        from .conv import Conv2DLayer
        from .pool import Pool2DLayer

    input_num_filters = l.output_shape[1]
    if increase_dim:
        first_stride = (2, 2)
        out_num_filters = input_num_filters * 2
    else:
        first_stride = (1, 1)
        out_num_filters = input_num_filters

    stack_1 = batch_norm(
        Conv2DLayer(l, num_filters=out_num_filters, filter_size=(3, 3),
            stride=first_stride, nonlinearity=nonlinearities.rectify,
            pad='same', W=init.HeNormal(gain='relu'))
    )
    stack_2 = batch_norm(
        Conv2DLayer(stack_1, num_filters=out_num_filters, filter_size=(3, 3),
            stride=(1, 1), nonlinearity=None,
            pad='same', W=init.HeNormal(gain='relu'))
    )

    # add shortcut connections
    if increase_dim:
        if projection:
            # projection shortcut, as option B in paper
            projection = batch_norm(
                Conv2DLayer(l, num_filters=out_num_filters, filter_size=(1, 1),
                    stride=(2, 2), nonlinearity=None, pad='same', b=None))
            block = NonlinearityLayer(
                ElemwiseSumLayer([stack_2, projection]), nonlinearity=nonlinearities.rectify)
        else:
            # identity shortcut, as option A in paper
            # we use a pooling layer to get identity with strides, since identity layers with stride don't exist in Lasagne
            identity = Pool2DLayer(l, pool_size=1, stride=(2, 2), mode='average_exc_pad')
            padding = PadLayer(identity, [out_num_filters // 4, 0, 0], batch_ndim=1)
            block = NonlinearityLayer(
                ElemwiseSumLayer([stack_2, padding]), nonlinearity=nonlinearities.rectify)
    else:
        block = NonlinearityLayer(ElemwiseSumLayer([stack_2, l]), nonlinearity=nonlinearities.rectify)

    return block
