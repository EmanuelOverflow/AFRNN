import torch.nn as nn
import torch.nn.utils.weight_norm

from fuzzy import DenseDefuzzy, MaxMin2d, MaxYager2d


class Conv2d_CIFAR10(nn.Module):
    def __init__(self, num_classes, tnorm_p=1., normalized_defuzzy=False, bias=False, constraint=None):
        super(Conv2d_CIFAR10, self).__init__()
        self.layer = nn.Conv2d(3, 8, kernel_size=3, bias=False)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=30*30*8, out_features=num_classes)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, inputs, **kwargs):
        x = self.layer(inputs)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.linear(x)
        return self.logsoftmax(x)


class Conv2d_CIFAR100(nn.Module):
    def __init__(self, num_classes, tnorm_p=1., normalized_defuzzy=False, bias=False, constraint=None):
        super(Conv2d_CIFAR100, self).__init__()
        self.layer = nn.Conv2d(3, 8, kernel_size=3, bias=False)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=30*30*8, out_features=num_classes)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, inputs, **kwargs):
        x = self.layer(inputs)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.linear(x)
        return self.logsoftmax(x)


class Conv2d_FashionMNIST(nn.Module):
    def __init__(self, num_classes, tnorm_p=1., normalized_defuzzy=False, bias=False, constraint=None):
        super(Conv2d_FashionMNIST, self).__init__()
        self.layer = nn.Conv2d(1, 8, kernel_size=3, bias=False)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=26*26*8, out_features=num_classes)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, inputs, **kwargs):
        x = self.layer(inputs)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.linear(x)
        return self.logsoftmax(x)


class Conv2d_ImageNet(nn.Module):
    def __init__(self, num_classes, tnorm_p=1., normalized_defuzzy=False, bias=False, constraint=None):
        super(Conv2d_ImageNet, self).__init__()
        self.layer = nn.Conv2d(3, 8, kernel_size=3, bias=False)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=222*222*8, out_features=num_classes)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, inputs, **kwargs):
        x = self.layer(inputs)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.linear(x)
        return self.logsoftmax(x)


class MaxMinAFRNN_Mnist(nn.Module):
    def __init__(self, num_classes, tnorm_p=1., normalized_defuzzy=False, bias=False, constraint=None):
        super(MaxMinAFRNN_Mnist, self).__init__()
        self.maxmin = MaxMin2d(in_channels=1, out_fuzzy_channels=8, kernel_size=3, bias=bias,
                               kernel_constraint=constraint)
        self.flatten = nn.Flatten()
        self.defuzzy = DenseDefuzzy(in_features=26*26*8, out_features=num_classes, normalization=normalized_defuzzy)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, inputs):
        x = self.maxmin(inputs)
        x = self.flatten(x)
        x = self.defuzzy(x)
        return self.logsoftmax(x)


class MaxMinAFRNN_Cifar10(nn.Module):
    def __init__(self, num_classes, tnorm_p=1., normalized_defuzzy=False, bias=False, constraint=None):
        super(MaxMinAFRNN_Cifar10, self).__init__()
        self.fuzzy_layer = MaxMin2d(in_channels=3, out_fuzzy_channels=8, kernel_size=3, bias=bias,
                                    kernel_constraint=constraint)
        self.flatten = nn.Flatten()
        self.defuzzy = DenseDefuzzy(in_features=30*30*8, out_features=num_classes, normalization=normalized_defuzzy)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, inputs):
        x = self.fuzzy_layer(inputs)
        x = self.flatten(x)
        x = self.defuzzy(x)
        return self.logsoftmax(x)


class MaxMinAFRNN_CIFAR100(nn.Module):
    def __init__(self, num_classes, tnorm_p=1., normalized_defuzzy=False, bias=False, constraint=None):
        super(MaxMinAFRNN_CIFAR100, self).__init__()
        self.fuzzy_layer = MaxMin2d(in_channels=3, out_fuzzy_channels=8, kernel_size=3, bias=bias,
                                    kernel_constraint=constraint)
        self.flatten = nn.Flatten()
        self.defuzzy = DenseDefuzzy(in_features=30*30*8, out_features=num_classes, normalization=normalized_defuzzy)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, inputs):
        x = self.fuzzy_layer(inputs)
        x = self.flatten(x)
        x = self.defuzzy(x)
        return self.logsoftmax(x)


class MaxMinAFRNN_FashionMNIST(nn.Module):
    def __init__(self, num_classes, tnorm_p=1., normalized_defuzzy=False, bias=False, constraint=None):
        super(MaxMinAFRNN_FashionMNIST, self).__init__()
        self.fuzzy_layer = MaxMin2d(in_channels=1, out_fuzzy_channels=8, kernel_size=3, bias=bias,
                               kernel_constraint=constraint)
        self.flatten = nn.Flatten()
        self.defuzzy = DenseDefuzzy(in_features=26*26*8, out_features=num_classes, normalization=normalized_defuzzy)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, inputs):
        x = self.fuzzy_layer(inputs)
        x = self.flatten(x)
        x = self.defuzzy(x)
        return self.logsoftmax(x)


class MaxMin2LAFRNN_CIFAR10(nn.Module):
    def __init__(self, num_classes, tnorm_p=1., normalized_defuzzy=False, bias=False, constraint=None):
        super(MaxMin2LAFRNN_CIFAR10, self).__init__()
        self.fuzzy_layer = MaxMin2d(in_channels=3, out_fuzzy_channels=8, kernel_size=3, bias=bias,
                                    kernel_constraint=constraint)
        self.fuzzy_layer2 = MaxMin2d(in_channels=8, out_fuzzy_channels=16, kernel_size=3,
                                     bias=bias, kernel_constraint=constraint)
        self.flatten = nn.Flatten()
        self.defuzzy = DenseDefuzzy(in_features=28*28*16, out_features=num_classes, normalization=normalized_defuzzy)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, inputs, **kwargs):
        x = self.fuzzy_layer(inputs)
        x = self.fuzzy_layer2(x)
        x = self.flatten(x)
        x = self.defuzzy(x)
        return self.logsoftmax(x)


class MaxMin2LAFRNN_CIFAR100(nn.Module):
    def __init__(self, num_classes, tnorm_p=1., normalized_defuzzy=False, bias=False, constraint=None):
        super(MaxMin2LAFRNN_CIFAR100, self).__init__()
        self.fuzzy_layer = MaxMin2d(in_channels=3, out_fuzzy_channels=8, kernel_size=3, bias=bias,
                                    kernel_constraint=constraint)
        self.fuzzy_layer2 = MaxMin2d(in_channels=8, out_fuzzy_channels=16, kernel_size=3,
                                     bias=bias, kernel_constraint=constraint)
        self.flatten = nn.Flatten()
        self.defuzzy = DenseDefuzzy(in_features=28*28*16, out_features=num_classes, normalization=normalized_defuzzy)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, inputs, **kwargs):
        x = self.fuzzy_layer(inputs)
        x = self.fuzzy_layer2(x)
        x = self.flatten(x)
        x = self.defuzzy(x)
        return self.logsoftmax(x)


class MaxMin2LAFRNN_FashionMNIST(nn.Module):
    def __init__(self, num_classes, tnorm_p=1., normalized_defuzzy=False, bias=False, constraint=None):
        super(MaxMin2LAFRNN_FashionMNIST, self).__init__()
        self.fuzzy_layer = MaxMin2d(in_channels=1, out_fuzzy_channels=8, kernel_size=3, bias=bias,
                                    kernel_constraint=constraint)
        self.fuzzy_layer2 = MaxMin2d(in_channels=8, out_fuzzy_channels=16, kernel_size=3,
                                     bias=bias, kernel_constraint=constraint)
        self.flatten = nn.Flatten()
        self.defuzzy = DenseDefuzzy(in_features=24*24*16, out_features=num_classes, normalization=normalized_defuzzy)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, inputs, **kwargs):
        x = self.fuzzy_layer(inputs)
        x = self.fuzzy_layer2(x)
        x = self.flatten(x)
        x = self.defuzzy(x)
        return self.logsoftmax(x)


class MaxLukAFRNN_CIFAR10(nn.Module):
    def __init__(self, num_classes, tnorm_p=1., normalized_defuzzy=False, bias=False, constraint=None):
        super(MaxLukAFRNN_CIFAR10, self).__init__()
        self.fuzzy_layer = MaxYager2d(in_channels=3, out_fuzzy_channels=8, p=tnorm_p, kernel_size=3, bias=bias,
                                      kernel_constraint=constraint)
        self.flatten = nn.Flatten()
        self.defuzzy = DenseDefuzzy(in_features=30*30*8, out_features=num_classes, normalization=normalized_defuzzy)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, inputs):
        x = self.fuzzy_layer(inputs)
        x = self.flatten(x)
        x = self.defuzzy(x)
        return self.logsoftmax(x)


class MaxLukAFRNN_Cifar100(nn.Module):
    def __init__(self, num_classes, tnorm_p=1., normalized_defuzzy=False, bias=False, constraint=None):
        super(MaxLukAFRNN_Cifar100, self).__init__()
        self.fuzzy_layer = MaxYager2d(in_channels=3, out_fuzzy_channels=8, p=tnorm_p, kernel_size=3, bias=bias,
                                 kernel_constraint=constraint)
        self.flatten = nn.Flatten()
        self.defuzzy = DenseDefuzzy(in_features=30*30*8, out_features=num_classes, normalization=normalized_defuzzy)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, inputs):
        x = self.fuzzy_layer(inputs)
        x = self.flatten(x)
        x = self.defuzzy(x)
        return self.logsoftmax(x)


class MaxLukAFRNN_FashionMNIST(nn.Module):
    def __init__(self, num_classes, tnorm_p=1., normalized_defuzzy=False, bias=False, constraint=None):
        super(MaxLukAFRNN_FashionMNIST, self).__init__()
        self.fuzzy_layer = MaxYager2d(in_channels=1, out_fuzzy_channels=8, p=tnorm_p, kernel_size=3, bias=bias,
                                      kernel_constraint=constraint)
        self.flatten = nn.Flatten()
        self.defuzzy = DenseDefuzzy(in_features=26*26*8, out_features=num_classes, normalization=normalized_defuzzy)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, inputs):
        x = self.fuzzy_layer(inputs)
        x = self.flatten(x)
        x = self.defuzzy(x)
        return self.logsoftmax(x)


class MaxLukAFRNN_ImageNet(nn.Module):
    def __init__(self, num_classes, tnorm_p=1., normalized_defuzzy=False, bias=False, constraint=None):
        super(MaxLukAFRNN_ImageNet, self).__init__()
        self.fuzzy_layer = MaxYager2d(in_channels=3, out_fuzzy_channels=8, p=tnorm_p, kernel_size=3, bias=bias,
                                      kernel_constraint=constraint)
        self.flatten = nn.Flatten()
        self.defuzzy = DenseDefuzzy(in_features=222*222*8, out_features=num_classes, normalization=normalized_defuzzy)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, inputs):
        x = self.fuzzy_layer(inputs)
        x = self.flatten(x)
        x = self.defuzzy(x)
        return self.logsoftmax(x)


class MaxLuk2LAFRNN_Cifar10(nn.Module):
    def __init__(self, num_classes, tnorm_p=1., normalized_defuzzy=False, bias=False, constraint=None):
        super(MaxLuk2LAFRNN_Cifar10, self).__init__()
        self.fuzzy_layer = MaxYager2d(in_channels=3, out_fuzzy_channels=8, p=tnorm_p, kernel_size=3, bias=bias,
                                      kernel_constraint=constraint)
        self.fuzzy_layer2 = MaxYager2d(in_channels=8, out_fuzzy_channels=16, kernel_size=3,
                                       bias=bias, kernel_constraint=constraint)
        self.flatten = nn.Flatten()
        self.defuzzy = DenseDefuzzy(in_features=28*28*16, out_features=num_classes, normalization=normalized_defuzzy)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, inputs, **kwargs):
        x = self.fuzzy_layer(inputs)
        x = self.fuzzy_layer2(x)
        x = self.flatten(x)
        x = self.defuzzy(x)
        return self.logsoftmax(x)


class MaxLuk2LAFRNN_CIFAR100(nn.Module):
    def __init__(self, num_classes, tnorm_p=1., normalized_defuzzy=False, bias=False, constraint=None):
        super(MaxLuk2LAFRNN_CIFAR100, self).__init__()
        self.fuzzy_layer = MaxYager2d(in_channels=3, out_fuzzy_channels=8, p=tnorm_p, kernel_size=3, bias=bias,
                                      kernel_constraint=constraint)
        self.fuzzy_layer2 = MaxYager2d(in_channels=8, out_fuzzy_channels=16, kernel_size=3,
                                       bias=bias, kernel_constraint=constraint)
        self.flatten = nn.Flatten()
        self.defuzzy = DenseDefuzzy(in_features=28*28*16, out_features=num_classes, normalization=normalized_defuzzy)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, inputs, **kwargs):
        x = self.fuzzy_layer(inputs)
        x = self.fuzzy_layer2(x)
        x = self.flatten(x)
        x = self.defuzzy(x)
        return self.logsoftmax(x)


class MaxLuk2LAFRNN_FashionMNIST(nn.Module):
    def __init__(self, num_classes, tnorm_p=1., normalized_defuzzy=False, bias=False, constraint=None):
        super(MaxLuk2LAFRNN_FashionMNIST, self).__init__()
        self.fuzzy_layer = MaxYager2d(in_channels=1, out_fuzzy_channels=8, p=tnorm_p, kernel_size=3, bias=bias,
                                      kernel_constraint=constraint)
        self.fuzzy_layer2 = MaxYager2d(in_channels=8, out_fuzzy_channels=16, kernel_size=3,
                                       bias=bias, kernel_constraint=constraint)
        self.flatten = nn.Flatten()
        self.defuzzy = DenseDefuzzy(in_features=24*24*16, out_features=num_classes, normalization=normalized_defuzzy)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, inputs, **kwargs):
        x = self.fuzzy_layer(inputs)
        x = self.fuzzy_layer2(x)
        x = self.flatten(x)
        x = self.defuzzy(x)
        return self.logsoftmax(x)


class MaxLuk3LAFRNN_CIFAR10(nn.Module):
    def __init__(self, num_classes, tnorm_p=1., normalized_defuzzy=False, bias=False, constraint=None):
        super(MaxLuk3LAFRNN_CIFAR10, self).__init__()
        self.fuzzy_layer = MaxYager2d(in_channels=3, out_fuzzy_channels=8, p=tnorm_p, kernel_size=3, bias=bias,
                                      kernel_constraint=constraint)
        self.fuzzy_layer2 = MaxYager2d(in_channels=8, out_fuzzy_channels=16, kernel_size=3,
                                       bias=bias, kernel_constraint=constraint)
        self.fuzzy_layer3 = MaxYager2d(in_channels=16, out_fuzzy_channels=32, kernel_size=3,
                                       bias=bias, kernel_constraint=constraint)
        self.flatten = nn.Flatten()
        self.defuzzy = DenseDefuzzy(in_features=26*26*32, out_features=num_classes, normalization=normalized_defuzzy)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, inputs, **kwargs):
        x = self.fuzzy_layer(inputs)
        x = self.fuzzy_layer2(x)
        x = self.fuzzy_layer3(x)
        x = self.flatten(x)
        x = self.defuzzy(x)
        return self.logsoftmax(x)


class MaxLukMaxMinAFRNN_Cifar10(nn.Module):
    def __init__(self, num_classes, tnorm_p=1., normalized_defuzzy=False, bias=False, constraint=None):
        super(MaxLukMaxMinAFRNN_Cifar10, self).__init__()
        self.fuzzy_layer = MaxYager2d(in_channels=3, out_fuzzy_channels=8, p=tnorm_p, kernel_size=3, bias=bias,
                                      kernel_constraint=constraint)
        self.fuzzy_layer2 = MaxMin2d(in_channels=8, out_fuzzy_channels=16, kernel_size=3,
                                     bias=bias, kernel_constraint=constraint)
        self.flatten = nn.Flatten()
        self.defuzzy = DenseDefuzzy(in_features=28*28*16, out_features=num_classes, normalization=normalized_defuzzy)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, inputs, **kwargs):
        x = self.fuzzy_layer(inputs)
        x = self.fuzzy_layer2(x)
        x = self.flatten(x)
        x = self.defuzzy(x)
        return self.logsoftmax(x)


class MaxLukMaxMinAFRNN_CIFAR100(nn.Module):
    def __init__(self, num_classes, tnorm_p=1., normalized_defuzzy=False, bias=False, constraint=None):
        super(MaxLukMaxMinAFRNN_CIFAR100, self).__init__()
        self.fuzzy_layer = MaxYager2d(in_channels=3, out_fuzzy_channels=8, p=tnorm_p, kernel_size=3, bias=bias,
                                      kernel_constraint=constraint)
        self.fuzzy_layer2 = MaxMin2d(in_channels=8, out_fuzzy_channels=16, kernel_size=3,
                                     bias=bias, kernel_constraint=constraint)
        self.flatten = nn.Flatten()
        self.defuzzy = DenseDefuzzy(in_features=28*28*16, out_features=num_classes, normalization=normalized_defuzzy)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, inputs, **kwargs):
        x = self.fuzzy_layer(inputs)
        x = self.fuzzy_layer2(x)
        x = self.flatten(x)
        x = self.defuzzy(x)
        return self.logsoftmax(x)


class MaxLukMaxMinAFRNN_FashionMNIST(nn.Module):
    def __init__(self, num_classes, tnorm_p=1., normalized_defuzzy=False, bias=False, constraint=None):
        super(MaxLukMaxMinAFRNN_FashionMNIST, self).__init__()
        self.fuzzy_layer = MaxYager2d(in_channels=1, out_fuzzy_channels=8, p=tnorm_p, kernel_size=3, bias=bias,
                                      kernel_constraint=constraint)
        self.fuzzy_layer2 = MaxMin2d(in_channels=8, out_fuzzy_channels=16, kernel_size=3,
                                     bias=bias, kernel_constraint=constraint)
        self.flatten = nn.Flatten()
        self.defuzzy = DenseDefuzzy(in_features=24*24*16, out_features=num_classes, normalization=normalized_defuzzy)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, inputs, **kwargs):
        x = self.fuzzy_layer(inputs)
        x = self.fuzzy_layer2(x)
        x = self.flatten(x)
        x = self.defuzzy(x)
        return self.logsoftmax(x)


class MaxLearnableYagerAFRNN_Cifar10(nn.Module):
    def __init__(self, num_classes, tnorm_p=1., normalized_defuzzy=False, bias=False, constraint=None):
        super(MaxLearnableYagerAFRNN_Cifar10, self).__init__()
        self.fuzzy_layer = MaxYager2d(in_channels=3, out_fuzzy_channels=8, p=tnorm_p, kernel_size=3, bias=bias,
                                      kernel_constraint=constraint)
        self.flatten = nn.Flatten()
        self.defuzzy = DenseDefuzzy(in_features=30*30*8, out_features=num_classes, normalization=normalized_defuzzy)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, inputs, **kwargs):
        x = self.fuzzy_layer(inputs)
        # if 'writer' in kwargs:
        #     print('writer')
        #     kwargs['writer'].add_histogram('Fuzzy_layer/act', x[0])
        #     kwargs['writer'].add_histogram('Fuzzy_layer/weights', self.fuzzy_layer.weight.data)
        #     kwargs['writer'].add_scalar('Params/p', self.fuzzy_layer.p, kwargs['epoch'])
        x = self.flatten(x)
        x = self.defuzzy(x)
        # if 'writer' in kwargs:
        #     kwargs['writer'].add_histogram('Defuzzy/act', x[0])
        #     kwargs['writer'].add_histogram('Defuzzy/weights', self.defuzzy.function.weight.data)
        #     kwargs['writer'].add_histogram('Defuzzy/bias', self.defuzzy.function.bias.data)
        return self.logsoftmax(x)


class LeNet5AFRNN_CIFAR10(nn.Module):
    def __init__(self, num_classes, tnorm_p=1., normalized_defuzzy=False, bias=False, constraint=None):
        super(LeNet5AFRNN_CIFAR10, self).__init__()
        self.fuzzy_layer1 = MaxYager2d(in_channels=3, out_fuzzy_channels=6, p=tnorm_p, kernel_size=5, bias=bias,
                                       kernel_constraint=constraint)
        # self.maxpool1 = nn.MaxPool2d(2, 2)
        self.fuzzy_layer2 = MaxYager2d(in_channels=6, out_fuzzy_channels=16, p=tnorm_p, kernel_size=5, bias=bias,
                                       kernel_constraint=constraint)
        # self.maxpool2 = nn.MaxPool2d(2, 2)
        self.fuzzy_layer3 = MaxYager2d(in_channels=16, out_fuzzy_channels=120, p=tnorm_p, kernel_size=24, bias=bias,
                                       kernel_constraint=constraint)
        self.flatten = nn.Flatten()
        self.defuzzy4 = DenseDefuzzy(in_features=120, out_features=84, normalization=normalized_defuzzy)
        self.fc5 = nn.Linear(in_features=84, out_features=num_classes)
        self.out = nn.LogSoftmax(dim=-1)

    def forward(self, inputs):
        x = self.fuzzy_layer1(inputs)
        x = self.fuzzy_layer2(x)
        x = self.fuzzy_layer3(x)
        x = self.flatten(x)
        x = torch.tanh(self.defuzzy4(x))
        x = self.out(self.fc5(x))
        return x


class LeNet5SigAFRNN_CIFAR10(nn.Module):
    def __init__(self, num_classes, tnorm_p=1., normalized_defuzzy=False, bias=False, constraint=None):
        super(LeNet5SigAFRNN_CIFAR10, self).__init__()
        self.fuzzy_layer1 = MaxYager2d(in_channels=3, out_fuzzy_channels=6, p=tnorm_p, kernel_size=5, bias=bias,
                                       kernel_constraint=constraint)
        # self.maxpool1 = nn.MaxPool2d(2, 2)
        self.fuzzy_layer2 = MaxYager2d(in_channels=6, out_fuzzy_channels=16, p=tnorm_p, kernel_size=5, bias=bias,
                                       kernel_constraint=constraint)
        # self.maxpool2 = nn.MaxPool2d(2, 2)
        self.fuzzy_layer3 = MaxYager2d(in_channels=16, out_fuzzy_channels=120, p=tnorm_p, kernel_size=24, bias=bias,
                                       kernel_constraint=constraint)
        self.flatten = nn.Flatten()
        self.defuzzy4 = DenseDefuzzy(in_features=120, out_features=84, normalization=normalized_defuzzy)
        self.fc5 = nn.Linear(in_features=84, out_features=num_classes)
        self.out = nn.LogSoftmax(dim=-1)

    def forward(self, inputs):
        x = self.fuzzy_layer1(inputs)
        x = self.fuzzy_layer2(x)
        x = self.fuzzy_layer3(x)
        x = self.flatten(x)
        x = torch.sigmoid(self.defuzzy4(x))
        x = self.out(self.fc5(x))
        return x


class LeNet5_2AFRNN_CIFAR10(nn.Module):
    def __init__(self, num_classes, tnorm_p=1., normalized_defuzzy=False, bias=False, constraint=None):
        super(LeNet5_2AFRNN_CIFAR10, self).__init__()
        self.fuzzy_layer1 = MaxYager2d(in_channels=3, out_fuzzy_channels=6, p=tnorm_p, kernel_size=5, bias=bias,
                                       kernel_constraint=constraint)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.fuzzy_layer2 = MaxYager2d(in_channels=6, out_fuzzy_channels=16, p=tnorm_p, kernel_size=5, bias=bias,
                                       kernel_constraint=constraint)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.fuzzy_layer3 = MaxYager2d(in_channels=16, out_fuzzy_channels=120, p=tnorm_p, kernel_size=5, bias=bias,
                                       kernel_constraint=constraint)
        self.flatten = nn.Flatten()
        self.defuzzy4 = DenseDefuzzy(in_features=120, out_features=84, normalization=normalized_defuzzy)
        self.fc5 = nn.Linear(in_features=84, out_features=num_classes)
        self.out = nn.LogSoftmax(dim=-1)

    def forward(self, inputs):
        x = self.maxpool1(self.fuzzy_layer1(inputs))
        x = self.maxpool2(self.fuzzy_layer2(x))
        x = self.fuzzy_layer3(x)
        x = self.flatten(x)
        x = torch.tanh(self.defuzzy4(x))
        x = self.out(self.fc5(x))
        return x


class LeNet5_2SigAFRNN_CIFAR10(nn.Module):
    def __init__(self, num_classes, tnorm_p=1., normalized_defuzzy=False, bias=False, constraint=None):
        super(LeNet5_2SigAFRNN_CIFAR10, self).__init__()
        self.fuzzy_layer1 = MaxYager2d(in_channels=3, out_fuzzy_channels=6, p=tnorm_p, kernel_size=5, bias=bias,
                                       kernel_constraint=constraint)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.fuzzy_layer2 = MaxYager2d(in_channels=6, out_fuzzy_channels=16, p=tnorm_p, kernel_size=5, bias=bias,
                                       kernel_constraint=constraint)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.fuzzy_layer3 = MaxYager2d(in_channels=16, out_fuzzy_channels=120, p=tnorm_p, kernel_size=5, bias=bias,
                                       kernel_constraint=constraint)
        self.flatten = nn.Flatten()
        self.defuzzy4 = DenseDefuzzy(in_features=120, out_features=84, normalization=normalized_defuzzy)
        self.fc5 = nn.Linear(in_features=84, out_features=num_classes)
        self.out = nn.LogSoftmax(dim=-1)

    def forward(self, inputs):
        x = self.maxpool1(self.fuzzy_layer1(inputs))
        x = self.maxpool2(self.fuzzy_layer2(x))
        x = self.fuzzy_layer3(x)
        x = self.flatten(x)
        x = torch.sigmoid(self.defuzzy4(x))
        x = self.out(self.fc5(x))
        return x


class LeNet5_CIFAR10(nn.Module):
    def __init__(self, num_classes):
        super(LeNet5_CIFAR10, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
        self.flatten = nn.Flatten()
        self.fc4 = nn.Linear(in_features=120, out_features=84)
        self.fc5 = nn.Linear(in_features=84, out_features=num_classes)
        self.out = nn.LogSoftmax(dim=-1)

    def forward(self, inputs):
        x = torch.tanh(self.maxpool1(self.conv1(inputs)))
        x = torch.tanh(self.maxpool2(self.conv2(x)))
        x = torch.tanh(self.conv3(x))
        x = self.flatten(x)
        x = torch.tanh(self.fc4(x))
        x = self.out(self.fc5(x))
        return x


class LeNetLukAFRNN_CIFAR10(nn.Module):
    def __init__(self, num_classes, tnorm_p=1., normalized_defuzzy=False, bias=False, constraint=None):
        super(LeNetLukAFRNN_CIFAR10, self).__init__()
        self.fuzzy_layer1 = MaxYager2d(in_channels=3, out_fuzzy_channels=6, p=tnorm_p, kernel_size=5, bias=bias,
                                       kernel_constraint=constraint)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.fuzzy_layer2 = MaxYager2d(in_channels=6, out_fuzzy_channels=16, p=tnorm_p, kernel_size=5, bias=bias,
                                       kernel_constraint=constraint)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.fuzzy_layer3 = MaxYager2d(in_channels=16, out_fuzzy_channels=120, p=tnorm_p, kernel_size=5, bias=bias,
                                       kernel_constraint=constraint)
        self.flatten = nn.Flatten()
        self.defuzzy4 = DenseDefuzzy(in_features=120, out_features=num_classes, normalization=normalized_defuzzy)
        self.out = nn.LogSoftmax(dim=-1)

    def forward(self, inputs):
        x = self.maxpool1(self.fuzzy_layer1(inputs))
        x = self.maxpool2(self.fuzzy_layer2(x))
        x = self.fuzzy_layer3(x)
        x = self.flatten(x)
        x = self.defuzzy4(x)
        x = self.out(x)
        return x
