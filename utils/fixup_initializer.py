import jax.numpy as jnp
from jax import random
from jax import core
from jax._src import dtypes
from jax._src.nn.initializers import _compute_fans

from typing import Any, Protocol, Sequence, Union

ModuleDef = Any
KeyArray = random.KeyArray
Array = Any
DTypeLikeFloat = Any
DTypeLikeComplex = Any
DTypeLikeInexact = Any
RealNumeric = Any


class Initializer(Protocol):
  @staticmethod
  def __call__(key: KeyArray,
               shape: core.Shape,
               dtype: DTypeLikeInexact = jnp.float_) -> Array:
      ...


def fixup(l: int,
          m: int,
          in_axis: Union[int, Sequence[int]] = -2,
          out_axis: Union[int, Sequence[int]] = -1,
          batch_axis: Sequence[int] = (),
          dtype: DTypeLikeInexact = jnp.float_
          ) -> Initializer:
    """

    Parameters
    ----------
    l: int
    The number of blocks in the neural network.
    m: int
    The number of layers inside each residual block.
    in_axis: Union[int, Sequence[int]]
    Axis or sequence of axes of the input dimension in the weights
      array.
    out_axis: Union[int, Sequence[int]]
    Axis or sequence of axes of the output dimension in the weights
      array.
    batch_axis: Sequence[int]
    Axis or sequence of axes in the weight array that should be
      ignored.
    dtype: DTypeLikeInexact
    The dtype of the weights.

    Returns
    -------
    init: Initializer
    An initializer for the parameter group.

    """
    if not isinstance(l, int) or l < 0:
        raise ValueError("The number of blocks in the network 'l' has to be a positive integer.")
    if not isinstance(m, int) or (2*m-2) <= 0:
        raise ValueError("The number of layers 'm' per block should be an integer m>=2.")

    def init(key: KeyArray,
           shape: core.Shape,
           dtype: DTypeLikeInexact = dtype,
           ) -> Array:

        fixup_scale = l**(-1/(2*m-2))

        dtype = dtypes.canonicalize_dtype(dtype)
        named_shape = core.as_named_shape(shape)
        fan_in, fan_out = _compute_fans(named_shape, in_axis, out_axis, batch_axis)
        variance = jnp.array(2 / fan_in, dtype=dtype)

        return random.normal(key, named_shape, dtype) * jnp.sqrt(variance)* fixup_scale

    return init
