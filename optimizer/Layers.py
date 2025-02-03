from abc import ABC, abstractmethod
import torch.nn as nn
import random
import copy


class Layer(ABC):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    @abstractmethod
    def calc_output_shape(self):
        pass

    @abstractmethod
    def to_nn_layer(self):
        pass

    @abstractmethod
    def crossover(self, other):
        pass


class LinearLayer(Layer):
    def __init__(self, input_shape, out_features):
        self.out_features = out_features
        super().__init__(input_shape)

    def calc_output_shape(self):
        return (self.out_features,)

    def to_nn_layer(self):
        in_features = self.input_shape[0] if len(self.input_shape) == 1 else self.input_shape[0] * self.input_shape[1] * \
                                                                             self.input_shape[2]
        return nn.Linear(in_features, self.out_features)

    def crossover(self, other):
        if not isinstance(other, LinearLayer):
            raise ValueError("Crossover must be performed with same layer type")
        new_out_features = random.choice([self.out_features, other.out_features])
        return LinearLayer(self.input_shape, new_out_features)


class Conv2dLayer(Layer):
    def __init__(self, input_shape, out_channels, kernel_size, stride=1, padding=0):
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        super().__init__(input_shape)

    def calc_output_shape(self):
        if(len(self.input_shape) < 3):
            print("bad input size conv2d")
        h = ((self.input_shape[1] + 2 * self.padding - self.kernel_size) // self.stride) + 1
        w = ((self.input_shape[2] + 2 * self.padding - self.kernel_size) // self.stride) + 1
        return (self.out_channels, h, w)

    def to_nn_layer(self):
        return nn.Conv2d(self.input_shape[0], self.out_channels, self.kernel_size,
                         stride=self.stride, padding=self.padding)

    def crossover(self, other):
        if not isinstance(other, Conv2dLayer):
            raise ValueError("Crossover must be performed with same layer type")
        new_out_channels = random.choice([self.out_channels, other.out_channels])
        new_kernel_size = random.choice([self.kernel_size, other.kernel_size])
        new_stride = random.choice([self.stride, other.stride])
        new_padding = random.choice([self.padding, other.padding])
        return Conv2dLayer(self.input_shape, new_out_channels, new_kernel_size,
                           new_stride, new_padding)


class MaxPool2dLayer(Layer):
    def __init__(self, input_shape, kernel_size, stride=None, padding=0):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        super().__init__(input_shape)

    def calc_output_shape(self):
        h = ((self.input_shape[1] + 2 * self.padding - self.kernel_size) // self.stride) + 1
        w = ((self.input_shape[2] + 2 * self.padding - self.kernel_size) // self.stride) + 1
        return (self.input_shape[0], h, w)

    def to_nn_layer(self):
        return nn.MaxPool2d(self.kernel_size, stride=self.stride, padding=self.padding)

    def crossover(self, other):
        if not isinstance(other, MaxPool2dLayer):
            raise ValueError("Crossover must be performed with same layer type")
        new_kernel_size = random.choice([self.kernel_size, other.kernel_size])
        new_stride = random.choice([self.stride, other.stride])
        new_padding = random.choice([self.padding, other.padding])
        return MaxPool2dLayer(self.input_shape, new_kernel_size, new_stride, new_padding)


class DropoutLayer(Layer):
    def __init__(self, input_shape, p=0.5):
        self.p = p
        super().__init__(input_shape)

    def calc_output_shape(self):
        return self.input_shape

    def to_nn_layer(self):
        return nn.Dropout(self.p)

    def crossover(self, other):
        if not isinstance(other, DropoutLayer):
            raise ValueError("Crossover must be performed with same layer type")
        new_p = random.choice([self.p, other.p])
        return DropoutLayer(self.input_shape, new_p)


class ReluLayer(Layer):
    def calc_output_shape(self):
        return self.input_shape

    def to_nn_layer(self):
        return nn.ReLU()

    def crossover(self, other):
        if not isinstance(other, ReluLayer):
            raise ValueError("Crossover must be performed with same layer type")
        return ReluLayer(self.input_shape)
