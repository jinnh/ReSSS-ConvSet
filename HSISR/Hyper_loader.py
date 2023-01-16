import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable
import scipy.io as scio
import torch.nn.functional as F
import random
from option import opt
import cv2

new_load = lambda *a, **k: np.load(*a, allow_pickle=True, **k)

class Hyper_dataset(Dataset):
    """
    get the high resolution and low resolution Hyperspectral images
    """
    def __init__(self, output_shape=512, ratio=1, Training_mode='Train', data_name='CAVE', use_generated_data=False,
                 use_all_data=True):
        self.data_name = data_name

        if data_name == 'CAVE':

            # training and testing mat files
            self.path = './datasets/HSI/CAVE/TrainTestMAT/' + str(opt.upscale_factor) + '/'

            # training and testing filename
            name = scio.loadmat("./datasets/HSI/CAVE/cave_train_test_filename.mat")

            self.train_name = name['train']
            self.test_name = name['test']
            self.num_pre_img = len(name['test'])
            self.train_len = len(name['train']) * len(name['test']) * len(name['test'])
            self.test_len = len(name['test'])
        elif data_name == 'Harvard':
            self.path = './datasets/HSI/Harvard/TrainTestMat_256/'+str(opt.upscale_factor)+'/'

            name = scio.loadmat("./datasets/HSI/Harvard/harvard_train_test_filename.mat")
            self.train_name = name['train']
            self.test_name = name['test']
            self.num_width = int(256/64) - 1
            self.num_height = int(256/64) - 1
            self.train_len = len(name['train']) * self.num_width * self.num_height
            self.test_len = len(name['test'])
            
        self.TM = Training_mode

    def __len__(self):
        if self.TM == 'Train':
            return self.train_len
        elif self.TM == 'Test':
            return self.test_len

    def zoom_img(self, input_img, ratio_):
        output_shape = int(input_img.shape[-1] * ratio_)
        return np.concatenate([self.zoom_img_(img, output_shape=output_shape)[np.newaxis, :, :] for img in input_img],
                              0)

    def zoom_img_(self, input_img, output_shape):
        return input_img.reshape(input_img.shape[0], output_shape, -1).mean(-1).swapaxes(0, 1).reshape(output_shape,
                                                                                                       output_shape,
                                                                                                       -1).mean(
            -1).swapaxes(0, 1)

    def recon_img(self, input_img):
        return cv2.resize(cv2.resize(input_img.transpose(1, 2, 0), dsize=(self.shape1, self.shape1)),
                          dsize=(self.output_shape, self.output_shape)).transpose(2, 0, 1)

    def __getitem__(self, index):

            if self.TM == 'Train':

                if self.data_name == 'CAVE' or self.data_name=='IEEE':
                    # if self.direct_data == True:
                    index_img = index // self.num_pre_img ** 2
                    # index_img = self.shuffle_index[index_img]-1
                    index_inside_image = index % self.num_pre_img ** 2
                    index_row = index_inside_image // self.num_pre_img
                    index_col = index_inside_image % self.num_pre_img

                elif self.data_name == 'Harvard':
                    index_img = index // (self.num_width * self.num_height)

                    index_inside_image = index % (self.num_width * self.num_height)
                    index_row = index_inside_image // self.num_height
                    index_col = index_inside_image % self.num_height

                hsi = scio.loadmat(self.path + str.rstrip(self.train_name[index_img]),verify_compressed_data_integrity=False)
                hsi_hr = hsi['HR']  # 512*512*31
                hsi_lr = hsi['LR']  # 128*128*31

                if (opt.upscale_factor == 4):
                    if(self.data_name=='Harvard' or self.data_name=='IEEE'):
                        hsi_hr = hsi_hr[index_row * 64:(index_row + 2) * 64, index_col * 64:(index_col + 2) * 64, :]  # 128*128*31
                        hsi_lr = hsi_lr[index_row * 16:(index_row + 2) * 16, index_col * 16:(index_col + 2) * 16, :]  # 32*32*31
                    elif(self.data_name=='CAVE'):
                        hsi_hr = hsi_hr[index_row * 32:(index_row + 4) * 32, index_col * 32:(index_col + 4) * 32,:]  # 128*128*31
                        hsi_lr = hsi_lr[index_row * 8:(index_row + 4) * 8, index_col * 8:(index_col + 4) * 8, :]  # 32*32*31
                elif (opt.upscale_factor == 8):
                    if(self.data_name=='Harvard' or self.data_name=='IEEE'):                                     
                        hsi_hr = hsi_hr[index_row * 64:(index_row + 2) * 64, index_col * 64:(index_col + 2) * 64, :]  # 128*128*31
                        hsi_lr = hsi_lr[index_row * 8:(index_row + 2) * 8, index_col * 8:(index_col + 2) * 8, :]  # 16*16*31
                    elif(self.data_name=='CAVE'):
                        hsi_hr = hsi_hr[index_row * 32:(index_row + 4) * 32, index_col * 32:(index_col + 4) * 32,:]  # 128*128*31
                        hsi_lr = hsi_lr[index_row * 4:(index_row + 4) * 4, index_col * 4:(index_col + 4) * 4, :]  # 16*16*31

                rotTimes = random.randint(0, 3)
                vFlip = random.randint(0, 1)
                hFlip = random.randint(0, 1)

                # Random rotation
                for j in range(rotTimes):
                    hsi_hr = np.rot90(hsi_hr)
                    hsi_lr = np.rot90(hsi_lr)

                # Random vertical Flip
                for j in range(vFlip):
                    hsi_hr = np.flip(hsi_hr, axis=1)
                    hsi_lr = np.flip(hsi_lr, axis=1)

                # Random Horizontal Flip
                for j in range(hFlip):
                    hsi_hr = np.flip(hsi_hr, axis=0)
                    hsi_lr = np.flip(hsi_lr, axis=0)

                hsi_hr = np.transpose(hsi_hr, (2, 0, 1)).astype(np.float32).copy()
                hsi_lr = np.transpose(hsi_lr, (2, 0, 1)).astype(np.float32).copy()

                return torch.from_numpy(hsi_lr), torch.from_numpy(hsi_hr)

            elif self.TM == 'Test':

                hsi = scio.loadmat(self.path + str.rstrip(str.rstrip(self.test_name[index])),
                                   verify_compressed_data_integrity=False)
                hsi_hr = hsi['HR']  # 512*512*31
                hsi_lr = hsi['LR']  # 128*128*31

                hsi_hr = np.transpose(hsi_hr, (2, 0, 1)).astype(np.float32)
                hsi_lr = np.transpose(hsi_lr, (2, 0, 1)).astype(np.float32)

                return torch.from_numpy(hsi_lr).float(), torch.from_numpy(hsi_hr).float()

