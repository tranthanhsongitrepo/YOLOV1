import torch
from torch import nn

config = [
    (7, 64, 2),
    "M",
    (3, 192, 1),
    "M",
    (1, 128, 1),
    (3, 256, 1),
    (1, 256, 1),
    (3, 512, 1),
    "M",
    [(1, 256, 1), (3, 512, 1), 4],
    (1, 512, 1),
    (3, 1024, 1),
    "M",
    [(1, 512, 1), (3, 1024, 1), 2],
    (3, 1024, 1),
    (3, 1024, 2),
    (3, 1024, 1),
    (3, 1024, 1),
]


def get_padding(shape):
    return int((shape - 1) / 2)


class CNNBlock(nn.Module):
    def __init__(self, in_channel, out_channel, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, bias=False, **kwargs)
        self.layer_norm = nn.BatchNorm2d(out_channel)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leaky_relu(self.layer_norm(self.conv(x)))


class YoloV1(nn.Module):
    def __init__(self, in_channel=3, grid_size=7, num_anchor=2, num_class=20):
        super(YoloV1, self).__init__()
        self.architecture = config
        self.in_channel = in_channel

        self.darknet = self.create_darknet()
        self.fc = self.create_fc(grid_size, num_anchor, num_class)

    def create_darknet(self):
        layers = []
        in_channel = self.in_channel
        for x in self.architecture:
            if x == "M":
                layers += [nn.MaxPool2d(2, 2)]
            elif type(x) == tuple:
                layers += [CNNBlock(in_channel, x[1], kernel_size=x[0], stride=x[2], padding=get_padding(x[0]))]
                in_channel = x[1]
            else:
                for _ in range(x[2]):
                    for item in x[:1]:
                        layers += [CNNBlock(in_channel, item[1], kernel_size=item[0], stride=item[2],
                                           padding=get_padding(item[0]))]
                        in_channel = item[1]
        return nn.Sequential(*layers)

    def create_fc(self, grid_size, num_anchor, num_class):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * grid_size * grid_size, 4096),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, grid_size * grid_size * (num_class + num_anchor * 5))
        )

    def forward(self, x):
        return self.fc(self.darknet(x))


def test(grid_size=7, num_anchor=2, num_class=20):
    model = YoloV1(grid_size=grid_size, num_anchor=num_anchor, num_class=num_class)
    return model(torch.rand((2, 3, 448, 448)))
