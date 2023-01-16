import os, sys
from os import listdir
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from option import opt
from utils.metrics import is_image_file, compare_mpsnr, compare_mssim, cal_sam, compare_rmse
import torch.nn.functional as F
import scipy.io as scio
from model.res3net import reconnetHRHSI as HSIG
from utils.serialization import Logger
from os import path as osp


def main():

    method = 'res3net'

    # your path of testing images
    input_path = '/datasets/HSI/' + opt.dataset + '/Test/' + str(opt.upscale_factor) + '/'
    out_path = 'results/' +  opt.dataset + '/'+ method +'/' + str(opt.upscale_factor) + '/'
    sys.stdout = Logger(osp.join(out_path, 'log_test.txt'))

    PSNRs = []
    SSIMs = []
    SAMs = []
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    if opt.cuda:
        print("=> use gpu id: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    model = HSIG()

    if opt.cuda:
        model = nn.DataParallel(model).cuda()

    checkpoint = torch.load(opt.checkpoint)

    model.load_state_dict(checkpoint["model"])
    images_name = [x for x in listdir(input_path) if is_image_file(x)]

    torch.cuda.empty_cache()
    with torch.no_grad():
        model.eval()
        for index in range(len(images_name)):

            mat = scio.loadmat(input_path + images_name[index])
            hyperLR = mat['LR'].transpose(2, 0, 1).astype(np.float32)
            input = Variable(torch.from_numpy(hyperLR).float(), volatile=True).contiguous().view(1, -1, hyperLR.shape[1],
                                                                                                    hyperLR.shape[2])
            if opt.cuda:
                input = input.cuda()

            SR, _ = model(input)

            SR = SR.cpu().numpy().squeeze(0).astype(np.float32)
            SR[SR < 0] = 0.
            SR[SR > 1.] = 1.

            HR = mat['HR'].transpose(2, 0, 1).astype(np.float32)
            
            psnr = compare_mpsnr(x_true=HR, x_pred=SR, data_range=1)
            ssim = compare_mssim(x_true=HR, x_pred=SR, data_range=1, multidimension=False)
            sam = cal_sam(SR, HR)


            PSNRs.append(psnr)
            SSIMs.append(ssim)
            SAMs.append(sam)


            SR = SR.transpose(1, 2, 0)
            HR = HR.transpose(1, 2, 0)

            scio.savemat(out_path + images_name[index], {'HR': HR, 'SR': SR})
            print("===The {}-th picture=====PSNR:{:.4f}=====SSIM:{:.4f}=====SAM:{:.4f}====Name:{}".format(index + 1, psnr,
                                                                                                         ssim, sam,
                                                                                                         images_name[index]))
        print("=====averPSNR:{:.4f}=====averSSIM:{:.4f}=====averSAM:{:.4f}====".format(np.mean(PSNRs), np.mean(SSIMs),
                                                                           np.mean(SAMs)))

if __name__ == "__main__":
    main()
