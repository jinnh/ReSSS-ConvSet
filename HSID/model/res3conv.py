import torch
import torch.nn as nn

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding=(0, 0, 0)):
        super(BasicConv, self).__init__()
        if(stride > 0):
            stride = (1, stride, stride)
            self.conv1 = nn.Conv3d(in_channel, out_channel,
                                     kernel_size=kernel_size, stride=stride,
                                     padding=padding, bias=False)
            self.conv2 = nn.Conv3d(out_channel, out_channel,
                                     kernel_size=kernel_size, stride=1,
                                     padding=padding, bias=False)
        else:
            stride = (1, abs(stride), abs(stride))
            self.conv1 = nn.ConvTranspose3d(in_channel, out_channel,
                                            kernel_size=kernel_size, stride=stride,
                                            padding=padding, output_padding=padding,bias=False)
            self.conv2 = nn.ConvTranspose3d(out_channel, out_channel,
                                            kernel_size=kernel_size, stride=1,
                                            padding=padding, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x


class ReConvSetBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super(ReConvSetBlock, self).__init__()

        self.reconvSet = nn.ModuleList([
            BasicConv(cin, cout, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
            BasicConv(cin, cout, kernel_size=(1, 3, 1), stride=1, padding=(0, 1, 0)),
            BasicConv(cin, cout, kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1))
        ])

        self.relu = nn.ReLU(inplace=True)
        self.aggregation1 = nn.Conv3d(cout * 3, cout, kernel_size=1, stride=1, padding=0, bias=False)
        self.aggregation2 = BasicConv(cout, cout, kernel_size=(1, 3, 3), stride=stride, padding=(0, 1, 1))

    def forward(self, x):

        hsi_features = []
        for layer in self.reconvSet:
            fea = layer(x)
            hsi_features.append(fea)
        hsi_features = torch.cat(hsi_features, dim=1)
        hsi_features = self.aggregation1(hsi_features)
        hsi_features = self.relu(hsi_features)
        output = self.aggregation2(hsi_features)

        return output

class RECEncoder(nn.Module):
    def __init__(self):
        super(RECEncoder, self).__init__()

        n_feats = 16
        kernel_size=3

        self.head = nn.Conv3d(1, n_feats, kernel_size, padding=kernel_size // 2, bias=False)
        self.encoderLayers = nn.ModuleList([
            ReConvSetBlock(cin=1 * n_feats, cout=n_feats * 1, stride=1),
            ReConvSetBlock(cin=1 * n_feats, cout=n_feats * 2, stride=2),
            ReConvSetBlock(cin=2 * n_feats, cout=n_feats * 2, stride=1),
            ReConvSetBlock(cin=2 * n_feats, cout=n_feats * 4, stride=2),
            ReConvSetBlock(cin=4 * n_feats, cout=n_feats * 4, stride=1),
            # ReConvSetBlock(cin=4 * n_feats, cout=n_feats * 8, stride=2),
            # ReConvSetBlock(cin=8 * n_feats, cout=n_feats * 8, stride=1)
        ])

    def forward(self, x, xu):

        x = self.head(x)
        xu.append(x)
        for i in range(len(self.encoderLayers)-1):
            x = self.encoderLayers[i](x)
            xu.append(x)
        x = self.encoderLayers[-1](x)

        return x

class RECDecoder(nn.Module):
    def __init__(self):
        super(RECDecoder, self).__init__()

        n_feats = 16
        kernel_size = 3

        self.decoderLayers = nn.ModuleList([
            # ReConvSetBlock(cin=8 * n_feats, cout=n_feats * 8, stride=1),
            # ReConvSetBlock(cin=8 * n_feats, cout=n_feats * 4, stride=-2),
            ReConvSetBlock(cin=4 * n_feats, cout=n_feats * 4, stride=1),
            ReConvSetBlock(cin=4 * n_feats, cout=n_feats * 2, stride=-2),
            ReConvSetBlock(cin=2 * n_feats, cout=n_feats * 2, stride=1),
            ReConvSetBlock(cin=2 * n_feats, cout=n_feats * 1, stride=-2),
            ReConvSetBlock(cin=1 * n_feats, cout=n_feats * 1, stride=1)
        ])

        self.tail = nn.Conv3d(n_feats, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x, xu):

        x = self.decoderLayers[0](x)
        for i in range(1, len(self.decoderLayers)):
            x = x + xu.pop()
            x = self.decoderLayers[i](x)
        x = x + xu.pop()
        x = self.tail(x)
        x = x + xu.pop()

        return x

class HSIDframework(nn.Module):
    def __init__(self):
        super(HSIDframework, self).__init__()

        self.encoder = RECEncoder()
        self.decoder = RECDecoder()

    def forward(self, x):

        x = x.unsqueeze(1)
        xu = [x]
        out = self.encoder(x, xu)
        out = self.decoder(out, xu)
        out = out.squeeze(1)

        return out
