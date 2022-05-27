import torch
import torch.nn as nn


class UNet(nn.Module):
    """Original U-Net architecture from Noise2Noise """

    def __init__(self, in_channels=3, out_channels=3):
        super(UNet, self).__init__()
        depthChannels= 48

        self._block1 = nn.Sequential(
            nn.Conv2d(in_channels, depthChannels, 3, stride=1, padding=1),
            #nn.BatchNorm2d(depthChannels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(depthChannels, depthChannels, 3, padding=1),
            #nn.BatchNorm2d(depthChannels),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2))

        self._block2 = nn.Sequential(
            nn.Conv2d(depthChannels, depthChannels, 3, stride=1, padding=1),
            #nn.BatchNorm2d(depthChannels),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2))
        self._block3 = nn.Sequential(
            nn.Conv2d(depthChannels, depthChannels, 3, stride=1, padding=1),
            #nn.BatchNorm2d(depthChannels),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2))
        self._block4 = nn.Sequential(
            nn.Conv2d(depthChannels, depthChannels, 3, stride=1, padding=1),
            #nn.BatchNorm2d(depthChannels),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2))
        self._block5 = nn.Sequential(
            nn.Conv2d(depthChannels, depthChannels, 3, stride=1, padding=1),
            #nn.BatchNorm2d(depthChannels),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2))

        self._block6 = nn.Sequential(
            nn.Conv2d(depthChannels, depthChannels, 3, stride=1, padding=1),
            #nn.BatchNorm2d(depthChannels),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(depthChannels, depthChannels, 3, stride=2, padding=1, output_padding=1))

        self._block7 = nn.Sequential(
            nn.Conv2d(depthChannels*2, depthChannels*2, 3, stride=1, padding=1),
            #nn.BatchNorm2d(depthChannels*2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(depthChannels*2, depthChannels*2, 3, stride=1, padding=1),
            #nn.BatchNorm2d(depthChannels * 2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(depthChannels*2, depthChannels*2, 3, stride=2, padding=1, output_padding=1))

        self._block8 = nn.Sequential(
            nn.Conv2d(depthChannels*3, depthChannels*2, 3, stride=1, padding=1),
            #nn.BatchNorm2d(depthChannels * 2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(depthChannels*2, depthChannels*2, 3, stride=1, padding=1),
            #nn.BatchNorm2d(depthChannels * 2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(depthChannels*2, depthChannels*2, 3, stride=2, padding=1, output_padding=1))

        self._block9 = nn.Sequential(
            nn.Conv2d(depthChannels*3, depthChannels*2, 3, stride=1, padding=1),
            #nn.BatchNorm2d(depthChannels * 2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(depthChannels*2, depthChannels*2, 3, stride=1, padding=1),
            #nn.BatchNorm2d(depthChannels * 2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(depthChannels*2, depthChannels*2, 3, stride=2, padding=1, output_padding=1))

        self._block10 = nn.Sequential(
            nn.Conv2d(depthChannels*3, depthChannels*2, 3, stride=1, padding=1),
            #nn.BatchNorm2d(depthChannels * 2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(depthChannels*2, depthChannels*2, 3, stride=1, padding=1),
            #nn.BatchNorm2d(depthChannels * 2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(depthChannels*2, depthChannels*2, 3, stride=2, padding=1, output_padding=1))

        self._block11 = nn.Sequential(
            nn.Conv2d(depthChannels*2 + in_channels, depthChannels*2, 3, stride=1, padding=1),
            #nn.BatchNorm2d(depthChannels * 2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(depthChannels*2, 32, 3, stride=1, padding=1),
            #nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, stride=1, padding=1),
            nn.ReLU())

        self._init_weights()

    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        # Encoder
        pool1 = self._block1(x)
        pool2 = self._block2(pool1)
        pool3 = self._block3(pool2)
        pool4 = self._block4(pool3)
        pool5 = self._block5(pool4)

        # Decoder
        upsample5 = self._block6(pool5)
        concat5 = torch.cat((upsample5, pool4), dim=1)
        upsample4 = self._block7(concat5)
        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self._block8(concat4)
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self._block9(concat3)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self._block10(concat2)
        concat1 = torch.cat((upsample1, x), dim=1)

        return self._block11(concat1)
