import torch
import torch.nn as nn
from numpy import linalg as la

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
        last_feat = x + xu.pop()

        x = self.tail(last_feat)
        x = x + xu.pop()
        return last_feat, x

class HSIDframework(nn.Module):
    def __init__(self):
        super(HSIDframework, self).__init__()

        self.encoder = RECEncoder()
        self.decoder = RECDecoder()

        self.downscale = nn.Conv2d(1, 1, kernel_size=(5, 5), stride=(4, 4), padding=(2, 2))
        # self.downscale = nn.Conv2d(1, 1, kernel_size=(9, 9), stride=(8, 8), padding=(4, 4))

    def svdLoss(self, x):

        [B, L, C, H, W] = x.shape
        recon_out_degradation = torch.reshape(x, [B*L*C, 1, H, W])
        recon_out_degradation = self.downscale(recon_out_degradation)
        recon_out_degradation = recon_out_degradation.reshape(
            [B, L, C, recon_out_degradation.shape[2], recon_out_degradation.shape[3]])
        last_feat = recon_out_degradation

        A = last_feat.reshape(
            (last_feat.shape[0], last_feat.shape[1], last_feat.shape[2] * last_feat.shape[3] * last_feat.shape[4]))

        U, Sigma, VT = torch.svd(A)

        Sigma_F2 = torch.norm(Sigma, dim=1)
        Sigma_F2 = Sigma_F2.unsqueeze(1)

        normalized_Sigma = Sigma / Sigma_F2

        Sigma_loss = torch.norm(normalized_Sigma, dim=1, p=1)
        Sigma_loss = torch.mean(Sigma_loss)

        return Sigma_loss

    def forward(self, x, mode):

        x = x.unsqueeze(1)
        xu = [x]
        out = self.encoder(x, xu)

        last_feat, out = self.decoder(out, xu)

        if mode == 'training':
            svd_loss = self.svdLoss(last_feat)
        else:
            svd_loss = 0

        out = out.squeeze(1)

        return out, svd_loss


if __name__ == "__main__":
    A = torch.rand(4, 64, 31, 16, 16)
