from flax import linen as nn
from functools import partial
from flax.linen.initializers import he_normal, zeros, ones
from typing import (Any, Callable, Tuple)
from utils.fixup_initializer import fixup

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any
Array = Any
ModuleDef = Any

default_kernel_init = he_normal()


class FixupBias(nn.Module):

    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        """
        Adds a scalar bias to the input.
        Parameters
        ----------
        inputs: Array
            The inputs to the fixup bias layer.

        Returns
        -------
        y: Array
            The ouputs of the bias layer.

        """

        bias = self.param('fixup_bias', self.bias_init, (1,))

        y = inputs+bias

        return y


class FixupMultiplier(nn.Module):

    multiplier_init: Callable[[PRNGKey, Shape, Dtype], Array] = ones

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        """
        Multiplies the input with a scalar.
        Parameters
        ----------
        inputs: Array
            The inputs of the fixup multiplier layer.

        Returns
        -------
        y: Array
            The outputs of the fixup multiplier layer.

        """
        multiplier = self.param('fixup_multiplier', self.multiplier_init, (1,))
        y = inputs * multiplier
        return y


class FixupBasicBlock(nn.Module):
    """
    Fixup Basic layer.
    This is the basic block of the Fixup architecture. The second convolutional layer and the Fixup Bias are initialized
    to 0. The first convolutional layer is initialized according to the Fixup initialization. The Fixup Multiplier is
    initialized to the value of 1.
    """
    in_planes: int
    out_planes: int
    stride: int
    dropRate: float
    total_num_basic_blocks: int

    @nn.compact
    def __call__(self, x, train):
        if not self.in_planes == self.out_planes:
            x = FixupBias()(x)
            x = nn.relu(x)
            out = nn.Conv(features=self.out_planes, strides=(self.stride, self.stride), use_bias=False,
                          kernel_size=(3, 3), kernel_init=fixup(l=self.total_num_basic_blocks, m=2))(x)
            out = FixupBias()(out)
            out = nn.relu(out)
            out = FixupBias()(out)
        else:
            out = FixupBias()(x)
            out = nn.relu(out)
            out = nn.Conv(features=self.out_planes, use_bias=False, kernel_size=(3, 3),
                          kernel_init=fixup(l=self.total_num_basic_blocks, m=2))(out)
            out = FixupBias()(out)
            out = nn.relu(out)
            out = FixupBias()(out)
        if self.dropRate > 0:
            out = nn.Dropout(self.dropRate, deterministic=not train)(out)

        out = nn.Conv(features=self.out_planes, strides=(1, 1), padding=1, use_bias=False, kernel_size=(3, 3),
                      kernel_init=zeros)(out)
        out = FixupMultiplier()(out)
        out = FixupBias()(out)

        if not self.in_planes == self.out_planes:
            x = nn.Conv(features=self.out_planes, strides=(self.stride, self.stride), kernel_size=(1, 1), padding=0,
                        use_bias=False)(x)
            return x + out
        else:
            return x + out


class FixupNetworkBlock(nn.Module):
    """
    Fixup Network block.
    This creates concatenations of Fixup Basic Blocks.
    """
    nb_layers: int
    in_planes: int
    out_planes: int
    block_cls: ModuleDef
    stride: int
    dropRate: float
    total_num_basic_blocks: int

    @nn.compact
    def __call__(self, x, train):
        for i in range(self.nb_layers):
            x = self.block_cls(i == 0 and self.in_planes or self.out_planes,
                               self.out_planes,
                               i == 0 and self.stride or 1,
                               self.dropRate,
                               self.total_num_basic_blocks)(x, train)
        return x


class FixupWideResNet(nn.Module):
    """
    Fixup Wide ResNet.
    The main class of the Fixup architecture. The final (classification) layer is initialized to 0.
    """
    depth: int
    widen_factor: int
    num_classes: int = 10
    dropRate: float = 0.0
    final_average_pooling: int = 8

    def setup(self):
        nChannels = [16, 16*self.widen_factor, 32*self.widen_factor, 64*self.widen_factor]
        assert ((self.depth-4) % 6 == 0)
        block = FixupBasicBlock
        m = (self.depth - 4) // 6 #This is the number of basic blocks per block.
        total_num_basic_blocks = m * 3 #Since there are 3 blocks, the total number of basic blocks is 3*n.

        self.conv1 = nn.Conv(features=nChannels[0], strides=(1, 1), use_bias=False, kernel_size=(3, 3),padding=1)
        self.bias1 = FixupBias()

        self.block1 = FixupNetworkBlock(m, nChannels[0], nChannels[1], block, 1, self.dropRate, total_num_basic_blocks)
        self.block2 = FixupNetworkBlock(m, nChannels[1], nChannels[2], block, 2, self.dropRate, total_num_basic_blocks)
        self.block3 = FixupNetworkBlock(m, nChannels[2], nChannels[3], block, 2, self.dropRate, total_num_basic_blocks)

        self.bias2 = FixupBias()
        self.relu1 = nn.relu
        self.fc = nn.Dense(features=self.num_classes, kernel_init=zeros, bias_init=zeros)

    def __call__(self, x, train):
        out = self.bias1(x)
        out = self.conv1(out)
        out = self.block1(out, train)
        out = self.block2(out, train)
        out = self.block3(out, train)
        out = self.relu1(out)
        out = nn.avg_pool(out, (self.final_average_pooling, self.final_average_pooling))
        out = out.reshape((-1))
        out = self.bias2(out)
        out = self.fc(out)
        return out


class LeNetStandard(nn.Module):
    """The LeNet model."""
    hidden: int
    num_classes: int

    @nn.compact
    def __call__(self, x, train):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((-1))
        x = nn.Dense(features=self.hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.num_classes)(x)
        return x


class MLP(nn.Module):
    """A simple MLP model with 3 non-linear layers."""
    hidden1: int
    hidden2: int
    hidden3: int
    num_classes: int

    @nn.compact
    def __call__(self, x, train):
        x = x.reshape((-1))
        x = nn.Dense(features=self.hidden1)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden2)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden3)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.num_classes)(x)
        return x


FixupWideResNet16 = partial(FixupWideResNet, depth=16, widen_factor=4, dropRate=0.3)
FixupWideResNet22 = partial(FixupWideResNet, depth=22, widen_factor=4, dropRate=0.3)
FixupWideResNet28 = partial(FixupWideResNet, depth=52, widen_factor=4, dropRate=0.3)
LeNet = partial(LeNetStandard, hidden=256)
MLP_Large = partial(MLP, hidden1=784, hidden2=500, hidden3=300)
MLP_Small = partial(MLP, hidden1=300, hidden2=200, hidden3=100)

# Used for testing
_FixupWideResNet10 = partial(FixupWideResNet, depth=10, widen_factor=1, num_classes=10, dropRate=0.3)

# Specifically for fashion mnist
FixupWideResNet22_SmallInputs = partial(FixupWideResNet, depth=22, widen_factor=4, dropRate=0.3,
                                        final_average_pooling=2)






