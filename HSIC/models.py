# -*- coding: utf-8 -*-
# Torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.nn import init


class HamidaEtAl(nn.Module):
    """
    3-D Deep Learning Approach for Remote Sensing Image Classification
    Amina Ben Hamida, Alexandre Benoit, Patrick Lambert, Chokri Ben Amar
    IEEE TGRS, 2018
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8344565
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=5, dilation=1):
        super(HamidaEtAl, self).__init__()
        # The first layer is a (3,3,3) kernel sized Conv characterized
        # by a stride equal to 1 and number of neurons equal to 20
        self.patch_size = patch_size
        self.input_channels = input_channels
        dilation = (dilation, 1, 1)

        if patch_size == 3:
            self.conv1 = nn.Conv3d(
                # 1, 20, (3, 3, 3), stride=(1, 1, 1), dilation=dilation, padding=1)
                1, 20, (3, 3, 3), stride=(1, 1, 1), padding=1)
        else:
            self.conv1 = nn.Conv3d(
                1, 20, (3, 3, 3), stride=(1, 1, 1), dilation=dilation, padding=0)
        # Next pooling is applied using a layer identical to the previous one
        # with the difference of a 1D kernel size (1,1,3) and a larger stride
        # equal to 2 in order to reduce the spectral dimension
        self.pool1 = nn.Conv3d(
            20, 20, (3, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0))
        # Then, a duplicate of the first and second layers is created with
        # 35 hidden neurons per layer.
        self.conv2 = nn.Conv3d(
            20, 35, (3, 3, 3), dilation=dilation, stride=(1, 1, 1), padding=(1, 1, 1))
        self.pool2 = nn.Conv3d(
            35, 35, (3, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0))

        # Finally, the 1D spatial dimension is progressively reduced
        # thanks to the use of two Conv layers, 35 neurons each,
        # with respective kernel sizes of (1,1,3) and (1,1,2) and strides
        # respectively equal to (1,1,1) and (1,1,2)
        self.conv3 = nn.Conv3d(
            35, 35, (3, 1, 1), dilation=dilation, stride=(1, 1, 1), padding=(1, 0, 0))
        self.conv4 = nn.Conv3d(
            35, 35, (2, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0))

        # self.dropout = nn.Dropout(p=0.5)

        self.features_size = self._get_final_flattened_size()
        # The architecture ends with a fully connected layer where the number
        # of neurons is equal to the number of input classes.
        self.fc = nn.Linear(self.features_size, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels,
                             self.patch_size, self.patch_size))
            x = self.pool1(self.conv1(x))
            x = self.pool2(self.conv2(x))
            x = self.conv3(x)
            x = self.conv4(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x, mode):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        # x = self.conv4(x)

        if mode == 'use_f1':
            last_feat = x
            A = last_feat.reshape(
                (last_feat.shape[0], last_feat.shape[1], last_feat.shape[2] * last_feat.shape[3] * last_feat.shape[4]))
            # print(A.shape)
            U, Sigma, VT = torch.svd(A)
            # print(Sigma[0])

            Sigma_F2 = torch.norm(Sigma, dim=1)
            Sigma_F2 = Sigma_F2.unsqueeze(1)
            # print(Sigma.shape, Sigma_F2.shape)

            normalized_Sigma = Sigma / Sigma_F2
            # print(normalized_Sigma[0])

            # Sigma_F2_loss = normalized_Sigma[:,0] - normalized_Sigma[:,63]
            Sigma_F1 = torch.norm(normalized_Sigma, dim=1, p=1)
            Sigma_F1_loss = torch.mean(Sigma_F1)

        x = x.view(-1, self.features_size)
        # x = self.dropout(x)
        x = self.fc(x)

        if mode == 'use_f1':
            return x, Sigma_F1_loss
        else:
            return x

class RE(nn.Module):
    """
    Our framework is constructed based on 3-D Deep Learning Approach for Remote Sensing Image Classification
    Amina Ben Hamida, Alexandre Benoit, Patrick Lambert, Chokri Ben Amar
    IEEE TGRS, 2018
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8344565
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=5, dilation=1):
        super(RE, self).__init__()
        # The first layer is a (3,3,3) kernel sized Conv characterized
        # by a stride equal to 1 and number of neurons equal to 20
        self.patch_size = patch_size
        self.input_channels = input_channels
        dilation = (dilation, 1, 1)

        if patch_size == 3:
            self.conv1 = nn.Conv3d(
                1, 20, (3, 3, 3), stride=(1, 1, 1), dilation=dilation, padding=0)
        else:
            self.conv1 = nn.Conv3d(
                1, 20, (3, 3, 3), stride=(1, 1, 1), dilation=dilation, padding=0)

        self.pool1 = nn.Conv3d(
            20, 20, (3, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0))

        self.conv2 = nn.ModuleList([
            nn.Conv3d(20, 35, (3, 1, 1), dilation=dilation, stride=(1, 1, 1), padding=(1, 0, 0)),
            nn.Conv3d(20, 35, (1, 3, 1), dilation=dilation, stride=(1, 1, 1), padding=(0, 1, 0)),
            nn.Conv3d(20, 35, (1, 1, 3), dilation=dilation, stride=(1, 1, 1), padding=(0, 0, 1))
        ])
        self.Conv_mixnas_2 = nn.Conv3d(35 * 3, 35, kernel_size=1, stride=1, padding=0)

        self.pool2 = nn.Conv3d(
            35, 35, (3, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0))

        self.conv3 = nn.Conv3d(
            35, 35, (3, 1, 1), dilation=dilation, stride=(1, 1, 1), padding=(1, 0, 0))
        self.conv4 = nn.Conv3d(
            35, 35, (2, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0))

        # self.dropout = nn.Dropout(p=0.5)

        self.features_size = self._get_final_flattened_size()
        self.fc = nn.Linear(self.features_size, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels,
                             self.patch_size, self.patch_size))
            x = F.relu(self.conv1(x))
            x = self.pool1(x)

            spectralnas = []
            for layer in self.conv2:
                nas_ = layer(x)
                spectralnas.append(nas_)
            spectralnas = torch.cat(spectralnas, dim=1)
            x = F.relu(self.Conv_mixnas_2(spectralnas))
            x = self.pool2(x)

            x = self.conv3(x)
            x = self.conv4(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x, mode='use_f1'):
        x = x.unsqueeze(1)

        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        spectralnas = []
        for layer in self.conv2:
            nas_ = layer(x)
            spectralnas.append(nas_)
        spectralnas = torch.cat(spectralnas, dim=1)
        x = F.relu(self.Conv_mixnas_2(spectralnas))

        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        if mode == 'use_f1':
  
            last_feat = x
            A = last_feat.reshape(
                (last_feat.shape[0], last_feat.shape[1], last_feat.shape[2] * last_feat.shape[3] * last_feat.shape[4]))
            U, Sigma, VT = torch.svd(A)

            Sigma_F2 = torch.norm(Sigma, dim=1)
            Sigma_F2 = Sigma_F2.unsqueeze(1)
            normalized_Sigma = Sigma / Sigma_F2
            Sigma_F1 = torch.norm(normalized_Sigma, dim=1, p=1)

            Sigma_F1_loss = torch.mean(Sigma_F1)

        else:
            Sigma_F1_loss = 0

        x = x.view(-1, self.features_size)

        x = self.fc(x)

        if mode == 'use_f1':
            return x, Sigma_F1_loss
        else:
            return x




def HamidaEtAl_3D(n_bands=1, n_classes=2, patch_size=1):
    return HamidaEtAl(n_bands, n_classes, patch_size)


def HamidaEtAl_RE(n_bands=1, n_classes=2, patch_size=1):
    return RE(n_bands, n_classes, patch_size)
