import numpy as np

import theano.tensor as T

from .base import MergeLayer
from .. import init, nonlinearities


__all__ = [
    "autocrop",
    "autocrop_array_shapes",
    "ConcatLayer",
    "concat",
    "ElemwiseMergeLayer",
    "ElemwiseSumLayer",
    "ExpressionMergeLayer",
    "ShortcutLayer"
]


def autocrop(inputs, cropping):
    """
    Crops the given input arrays.

    Cropping takes a sequence of inputs and crops them per-axis in order to
    ensure that their sizes are consistent so that they can be combined
    in an element-wise fashion. If cropping is enabled for a specific axis,
    the minimum size in that axis of all inputs is computed, and all
    inputs are cropped to that size.

    The per-axis cropping modes are:

    `None`: this axis is not cropped, inputs are unchanged in this axis

    `'lower'`: inputs are cropped choosing the lower portion in this axis
    (`a[:crop_size, ...]`)

    `'upper'`: inputs are cropped choosing the upper portion in this axis
    (`a[-crop_size:, ...]`)

    `'center'`: inputs are cropped choosing the central portion in this axis
    (``a[offset:offset+crop_size, ...]`` where
    ``offset = (a.shape[0]-crop_size)//2)``

    Parameters
    ----------
    inputs : list of Theano expressions
        The input arrays in the form of a list of Theano expressions

    cropping : list of cropping modes
        Cropping modes, one for each axis. If length of `cropping` is less
        than the number of axes in the inputs, it is padded with `None`.
        If `cropping` is None, `input` is returned as is.

    Returns
    -------
    list of Theano expressions
        each expression is the cropped version of the corresponding input

    Example
    -------
    For example, given three inputs:

    >>> import numpy
    >>> import theano

    >>> a = numpy.random.random((1, 2, 3, 4))
    >>> b = numpy.random.random((5, 4, 4, 2))
    >>> c = numpy.random.random((7, 1, 8, 9))

    Cropping mode for each axis:

    >>> cropping = [None, 'lower', 'center', 'upper']

    Crop (note that the input arrays are converted to Theano vars first,
    and that the results are converted back from Theano expressions to
    numpy arrays by calling `eval()`)
    >>> xa, xb, xc = autocrop([theano.shared(a), \
                               theano.shared(b), \
                               theano.shared(c)], cropping)
    >>> xa, xb, xc = xa.eval(), xb.eval(), xc.eval()

    They will be left as is in axis 0 and cropped in the other three,
    choosing the lower, center and upper portions:

    Axis 0: choose all, axis 1: lower 1 element,
    axis 2: central 3 (all) and axis 3: upper 2
    >>> (xa == a[:, :1, :3, -2:]).all()
    True

    Axis 0: choose all, axis 1: lower 1 element,
    axis 2: central 3 starting at 0 and axis 3: upper 2 (all)
    >>> (xb == b[:, :1, :3, -2:]).all()
    True

    Axis 0: all, axis 1: lower 1 element (all),
    axis 2: central 3 starting at 2 and axis 3: upper 2
    >>> (xc == c[:, :1, 2:5:, -2:]).all()
    True
    """
    if cropping is None:
        # No cropping in any dimension
        return inputs
    else:
        # Get the number of dimensions
        ndim = inputs[0].ndim
        # Check for consistent number of dimensions
        if not all(input.ndim == ndim for input in inputs):
            raise ValueError("Not all inputs are of the same "
                             "dimensionality. Got {0} inputs of "
                             "dimensionalities {1}.".format(
                                 len(inputs),
                                 [input.ndim for input in inputs]))
        # Get the shape of each input, where each shape will be a Theano
        # expression
        shapes = [input.shape for input in inputs]
        # Convert the shapes to a matrix expression
        shapes_tensor = T.as_tensor_variable(shapes)
        # Min along axis 0 to get the minimum size in each dimension
        min_shape = T.min(shapes_tensor, axis=0)

        # Nested list of slices; each list in `slices` corresponds to
        # an input and contains a slice for each dimension
        slices_by_input = [[] for i in range(len(inputs))]

        # If there are more dimensions than cropping entries, pad
        # the cropping
        cropping = list(cropping)
        if ndim > len(cropping):
            cropping = list(cropping) + \
                [None] * (ndim - len(cropping))

        # For each dimension
        for dim, cr in enumerate(cropping):
            if cr is None:
                # Don't crop this dimension
                slice_all = slice(None)
                for slices in slices_by_input:
                    slices.append(slice_all)
            else:
                # We crop all inputs in the dimension `dim` so that they
                # are the minimum found in this dimension from all inputs
                sz = min_shape[dim]
                if cr == 'lower':
                    # Choose the first `sz` elements
                    slc_lower = slice(None, sz)
                    for slices in slices_by_input:
                        slices.append(slc_lower)
                elif cr == 'upper':
                    # Choose the last `sz` elements
                    slc_upper = slice(-sz, None)
                    for slices in slices_by_input:
                        slices.append(slc_upper)
                elif cr == 'center':
                    # Choose `sz` elements from the center
                    for sh, slices in zip(shapes, slices_by_input):
                        offset = (sh[dim] - sz) // 2
                        slices.append(slice(offset, offset + sz))
                else:
                    raise ValueError(
                        'Unknown crop mode \'{0}\''.format(cr))

        return [input[slices] for input, slices in
                zip(inputs, slices_by_input)]


def autocrop_array_shapes(input_shapes, cropping):
    """
    Computes the shapes of the given arrays after auto-cropping is applied.

    For more information on cropping, see the :func:`autocrop` function
    documentation.

    Parameters
    ----------
    input_shapes : the shapes of input arrays prior to cropping in
        the form of a list of tuples

    cropping : a list of cropping modes, one for each axis. If length of
        `cropping` is less than the number of axes in the inputs, it is
        padded with `None`. If `cropping` is None, `input_shapes` is returned
        as is. For more information on their values and operation, see the
        :func:`autocrop` documentation.

    Returns
    -------
    list of tuples
        each tuple is a cropped version of the corresponding input
        shape tuple in `input_shapes`

    For example, given three input shapes with 4 axes each:

    >>> a = (1, 2, 3, 4)
    >>> b = (5, 4, 4, 2)
    >>> c = (7, 1, 8, 9)

    Cropping mode for each axis:

    >>> cropping = [None, 'lower', 'center', 'upper']

    Apply:

    >>> cropped_shapes = autocrop_array_shapes([a, b, c], cropping)
    >>> cropped_shapes[0]
    (1, 1, 3, 2)

    >>> cropped_shapes[1]
    (5, 1, 3, 2)

    >>> cropped_shapes[2]
    (7, 1, 3, 2)

    Note that axis 0 remains unchanged, where all the others are cropped
    to the minimum size in that axis.
    """
    if cropping is None:
        return input_shapes
    else:
        # Check for consistent number of dimensions
        ndim = len(input_shapes[0])
        if not all(len(sh) == ndim for sh in input_shapes):
            raise ValueError("Not all inputs are of the same "
                             "dimensionality. Got {0} inputs of "
                             "dimensionalities {1}.".format(
                                 len(input_shapes),
                                 [len(sh) for sh in input_shapes]))

        result = []

        # If there are more dimensions than cropping entries, pad
        # the cropping
        cropping = list(cropping)
        if ndim > len(cropping):
            cropping = list(cropping) + \
                [None] * (ndim - len(cropping))

        for sh, cr in zip(zip(*input_shapes), cropping):
            if cr is None:
                result.append(sh)
            elif cr in {'lower', 'center', 'upper'}:
                result.append([min(sh)] * len(sh))
            else:
                raise ValueError('Unknown crop mode \'{0}\''.format(cr))
        return [tuple(sh) for sh in zip(*result)]


class ConcatLayer(MergeLayer):

    """
    Concatenates multiple inputs along the specified axis. Inputs should have
    the same shape except for the dimension specified in axis, which can have
    different sizes.

    Parameters
    -----------
    incomings : a list of :class:`Layer` instances or tuples
        The layers feeding into this layer, or expected input shapes

    axis : int
        Axis which inputs are joined over

    cropping : None or [crop]
        Cropping for each input axis. Cropping is described in the docstring
        for :func:`autocrop`. Cropping is always disabled for `axis`.
    """

    def __init__(self, incomings, axis=1, cropping=None, **kwargs):
        super(ConcatLayer, self).__init__(incomings, **kwargs)
        self.axis = axis
        if cropping is not None:
            # If cropping is enabled, don't crop on the selected axis
            cropping = list(cropping)
            cropping[axis] = None
        self.cropping = cropping

    def get_output_shape_for(self, input_shapes):
        input_shapes = autocrop_array_shapes(input_shapes, self.cropping)
        # Infer the output shape by grabbing, for each axis, the first
        # input size that is not `None` (if there is any)
        output_shape = [next((s for s in sizes if s is not None), None)
                        for sizes in zip(*input_shapes)]

        def match(shape1, shape2):
            return (len(shape1) == len(shape2) and
                    all(i == self.axis or s1 is None or s2 is None or s1 == s2
                        for i, (s1, s2) in enumerate(zip(shape1, shape2))))

        # Check for compatibility with inferred output shape
        if not all(match(shape, output_shape) for shape in input_shapes):
            raise ValueError("Mismatch: input shapes must be the same except "
                             "in the concatenation axis")
        # Infer output shape on concatenation axis and return
        sizes = [input_shape[self.axis] for input_shape in input_shapes]
        concat_size = None if any(s is None for s in sizes) else sum(sizes)
        output_shape[self.axis] = concat_size
        return tuple(output_shape)

    def get_output_for(self, inputs, **kwargs):
        inputs = autocrop(inputs, self.cropping)
        return T.concatenate(inputs, axis=self.axis)

concat = ConcatLayer  # shortcut


class ElemwiseMergeLayer(MergeLayer):

    """
    This layer performs an elementwise merge of its input layers.
    It requires all input layers to have the same output shape.

    Parameters
    ----------
    incomings : a list of :class:`Layer` instances or tuples
        the layers feeding into this layer, or expected input shapes,
        with all incoming shapes being equal

    merge_function : callable
        the merge function to use. Should take two arguments and return the
        updated value. Some possible merge functions are ``theano.tensor``:
        ``mul``, ``add``, ``maximum`` and ``minimum``.

    cropping : None or [crop]
        Cropping for each input axis. Cropping is described in the docstring
        for :func:`autocrop`

    See Also
    --------
    ElemwiseSumLayer : Shortcut for sum layer.
    """

    def __init__(self, incomings, merge_function, cropping=None, **kwargs):
        super(ElemwiseMergeLayer, self).__init__(incomings, **kwargs)
        self.merge_function = merge_function
        self.cropping = cropping

    def get_output_shape_for(self, input_shapes):
        input_shapes = autocrop_array_shapes(input_shapes, self.cropping)
        # Infer the output shape by grabbing, for each axis, the first
        # input size that is not `None` (if there is any)
        output_shape = tuple(next((s for s in sizes if s is not None), None)
                             for sizes in zip(*input_shapes))

        def match(shape1, shape2):
            return (len(shape1) == len(shape2) and
                    all(s1 is None or s2 is None or s1 == s2
                        for s1, s2 in zip(shape1, shape2)))

        # Check for compatibility with inferred output shape
        if not all(match(shape, output_shape) for shape in input_shapes):
            raise ValueError("Mismatch: not all input shapes are the same")
        return output_shape

    def get_output_for(self, inputs, **kwargs):
        inputs = autocrop(inputs, self.cropping)
        output = None
        for input in inputs:
            if output is not None:
                output = self.merge_function(output, input)
            else:
                output = input
        return output


class ElemwiseSumLayer(ElemwiseMergeLayer):

    """
    This layer performs an elementwise sum of its input layers.
    It requires all input layers to have the same output shape.

    Parameters
    ----------
    incomings : a list of :class:`Layer` instances or tuples
        the layers feeding into this layer, or expected input shapes,
        with all incoming shapes being equal

    coeffs: list or scalar
        A same-sized list of coefficients, or a single coefficient that
        is to be applied to all instances. By default, these will not
        be included in the learnable parameters of this layer.

    cropping : None or [crop]
        Cropping for each input axis. Cropping is described in the docstring
        for :func:`autocrop`

    Notes
    -----
    Depending on your architecture, this can be used to avoid the more
    costly :class:`ConcatLayer`. For example, instead of concatenating layers
    before a :class:`DenseLayer`, insert separate :class:`DenseLayer` instances
    of the same number of output units and add them up afterwards. (This avoids
    the copy operations in concatenation, but splits up the dot product.)
    """

    def __init__(self, incomings, coeffs=1, cropping=None, **kwargs):
        super(ElemwiseSumLayer, self).__init__(incomings, T.add,
                                               cropping=cropping, **kwargs)
        if isinstance(coeffs, list):
            if len(coeffs) != len(incomings):
                raise ValueError("Mismatch: got %d coeffs for %d incomings" %
                                 (len(coeffs), len(incomings)))
        else:
            coeffs = [coeffs] * len(incomings)

        self.coeffs = coeffs

    def get_output_for(self, inputs, **kwargs):
        # if needed multiply each input by its coefficient
        inputs = [input * coeff if coeff != 1 else input
                  for coeff, input in zip(self.coeffs, inputs)]

        # pass scaled inputs to the super class for summing
        return super(ElemwiseSumLayer, self).get_output_for(inputs, **kwargs)

class ExpressionMergeLayer(MergeLayer):

    """
    This layer performs an custom expressions on list of inputs to merge them.
    This layer is different from ElemwiseMergeLayer by not required all
    input_shapes are equal

    Parameters
    ----------
    incomings : a list of :class:`Layer` instances or tuples
        the layers feeding into this layer, or expected input shapes

    merge_function : callable
        the merge function to use. Should take two arguments and return the
        updated value. Some possible merge functions are ``theano.tensor``:
        ``mul``, ``add``, ``maximum`` and ``minimum``.

    output_shape : None, callable, tuple, or 'auto'
        Specifies the output shape of this layer. If a tuple, this fixes the
        output shape for any input shape (the tuple can contain None if some
        dimensions may vary). If a callable, it should return the calculated
        output shape given the input shape. If None, the output shape is
        assumed to be the same as the input shape. If 'auto', an attempt will
        be made to automatically infer the correct output shape.

    Example
    --------
    >>> from lasagne.layers import InputLayer, DimshuffleLayer, ExpressionMergeLayer
    >>> l_in = lasagne.layers.InputLayer(shape=(None, 500, 120))
    >>> l_mask = lasagne.layers.InputLayer(shape=(None, 500))
    >>> l_dim = lasagne.layers.DimshuffleLayer(l_mask, pattern=(0, 1, 'x'))
    >>> l_out = lasagne.layers.ExpressionMergeLayer(
                                (l_in, l_dim), tensor.mul, output_shape='auto')
    (None, 500, 120)
    """

    def __init__(self, incomings, merge_function, output_shape=None, **kwargs):
        super(ExpressionMergeLayer, self).__init__(incomings, **kwargs)
        if output_shape is None:
            self._output_shape = None
        elif output_shape == 'auto':
            self._output_shape = 'auto'
        elif hasattr(output_shape, '__call__'):
            self.get_output_shape_for = output_shape
        else:
            self._output_shape = tuple(output_shape)

        self.merge_function = merge_function

    def get_output_shape_for(self, input_shapes):
        if self._output_shape is None:
            return input_shapes[0]
        elif self._output_shape is 'auto':
            input_shape = [(0 if s is None else s for s in ishape)
                           for ishape in input_shapes]
            Xs = [T.alloc(0, *ishape) for ishape in input_shape]
            output_shape = self.merge_function(*Xs).shape.eval()
            output_shape = tuple(s if s else None for s in output_shape)
            return output_shape
        else:
            return self._output_shape

    def get_output_for(self, inputs, **kwargs):
        return self.merge_function(*inputs)

class ShortcutLayer(MergeLayer):

    """
    Create shortcut tunnel for information flow by using simple combination of
    previous inputs:
     - With projection:
        y = F(x1) + W(2) * x2 + W(3) * x3 + ....
     - Without projection:
        y = F(x1) + x2 + x3 + ....

    Parameters
    ----------
    incomings : a list of :class:`Layer` instances or tuples
        the layers feeding into this layer, the first layer known as
        root and all other layers will be projected into that layer's output

    force_projection : bool
        whether projection will be used for all shortcut

    W : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the projection weights.
        These should be a matrix with shape ``(num_inputs, num_units)``.
        See :func:`lasagne.utils.create_param` for more information.

    nonlinearity : callable or None
        The nonlinearity that is applied to each shortcut. If None
        is provided, the layer will be linear.

    Example
    -------
    >>> ####### Highway network
    >>> from lasagne.layers import DenseLayer, InputLayer, ShortcutLayer
    >>> l_in = InputLayer(shape=(None, 1, 28, 28))
    >>> # project to 512-dim
    >>> l_in = DenseLayer(l_in, num_units=512)
    >>> # hidden layer
    >>> l_h = DenseLayer(l_in, num_units=512)
    >>> # gate layer
    >>> l_t = DenseLayer(incoming, num_units=512,
    >>>                  nonlinearity=T.nnet.sigmoid)
    >>> # (1 - T)
    >>> l_t_ = ExpressionLayer(l_t, lambda x: 1-x)
    >>> l = ShortcutLayer(
    >>>         ElemwiseMergeLayer((l_t,l_h),merge_function=T.mul),
    >>>         ElemwiseMergeLayer((l_t_,l_in),merge_function=T.mul)
    >>> )

    Notes
    -----
    Output_shape is the shape of first element in incomings array.
    If shortcut is reshape-able to fit the root's shape, not projection is
    performed.
    """

    def __init__(self, incomings, force_projection=False,
        W=init.GlorotUniform(), nonlinearity=nonlinearities.identity,
        **kwargs):
        super(ShortcutLayer, self).__init__(incomings, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)
        self.W = []

        root = incomings[0]
        root_outdim = int(np.prod(root.output_shape[1:]))
        # create projection if necessary
        for l in incomings[1:]:
            shortcut_outdim = int(np.prod(l.output_shape[1:]))
            if root_outdim != shortcut_outdim or force_projection:
                self.W.append(self.add_param(W, (shortcut_outdim, root_outdim), name="W"))
            else:
                self.W.append(None)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, **kwargs):
        activation = inputs[0]
        ndim = activation.ndim
        for W, x in zip(self.W, inputs[1:]):
            if W is not None: # projection
                if x.ndim > 2: # flatten > 2D input
                    x = x.flatten(2)
                x = T.dot(x, W)
            x = self.nonlinearity(x)
            if ndim > 2: # reshape for > 2D
                x = x.reshape(activation.shape)
            activation += x
        return activation
