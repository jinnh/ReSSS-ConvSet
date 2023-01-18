import os, sys
from os import listdir
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from option import opt
from metrics import is_image_file, compare_mpsnr, compare_mssim, cal_sam
import scipy.io as scio
from serialization import Logger
from model.res3net import HSIDframework as HSID
from os import path as osp

def main():

    method = 'Res3Net'
    GaussianNoiseType = ['icvl_512_30', 'icvl_512_50', 'icvl_512_70', 'icvl_512_blind']
    ComplexNoiseType = ['icvl_512_noniid', 'icvl_512_stripe', 'icvl_512_deadline', 'icvl_512_impulse','icvl_512_mixture']

    for noiseType in GaussianNoiseType:

        print('Testing...', noiseType)

        input_root = './datasets/ICVL/'
        output_root = './datasets/ICVL/results/' + noiseType + '/'
        input_path = input_root + noiseType + '/'
        if not os.path.exists(output_root):
            os.mkdir(output_root)
        sys.stdout = Logger(osp.join(output_root, noiseType + '_test_results.txt'))

        PSNRs = []
        SSIMs = []
        SAMs = []

        if opt.cuda:
            print("=> use gpu id: '{}'".format(opt.gpus))
            os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
            if not torch.cuda.is_available():
                raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

        model = HSID()

        if opt.cuda:
            model = nn.DataParallel(model).cuda()

        checkpoint = torch.load(opt.checkpoint)
        model.load_state_dict(checkpoint["model"])

        images_name = [x for x in listdir(input_path) if is_image_file(x)]

        for i in range(1):
            with torch.no_grad():
                model.eval()
                for index in range(len(images_name)):

                    torch.cuda.empty_cache()
                    mat = scio.loadmat(input_path + images_name[index])
                    input = mat['input'].transpose(2, 0, 1).astype(np.float32)

                    input = Variable(torch.from_numpy(input).float(), volatile=True).contiguous().view(1, -1, input.shape[1],
                                                                                                     input.shape[2])
                    if opt.cuda:
                        input = input.cuda()


                    output, _ = model(input, 'testing')

                    output = output.squeeze(0)
                    output = output.cpu().numpy().astype(np.float32)

                    output[output < 0] = 0
                    output[output > 1.] = 1.
                    Denoise = output

                    GT = mat['gt'].transpose(2, 0, 1).astype(np.float32)

                    psnr = compare_mpsnr(x_true=GT, x_pred=Denoise, data_range=1)
                    ssim = compare_mssim(x_true=GT, x_pred=Denoise, data_range=1, multidimension=False)
                    sam = cal_sam(GT, Denoise)


                    PSNRs.append(psnr)
                    SSIMs.append(ssim)
                    SAMs.append(sam)

                    Denoise = Denoise.transpose(1, 2, 0)
                    GT = GT.transpose(1, 2, 0)
                    out_path = output_root+images_name[index].split('.')[0] + '/'
                    if not os.path.exists(out_path):
                        os.mkdir(out_path)
                    scio.savemat(out_path + '/' + method + '.mat', {'GT': GT, 'R_hsi': Denoise})
                    print("===The {}-th picture=====PSNR:{:.4f}=====SSIM:{:.4f}=====SAM:{:.4f}=====Name:{}".format(index + 1, psnr,
                                                                                                                  ssim, sam,
                                                                                                                  images_name[
                                                                                                                      index], ))
            print("=====averPSNR:{:.4f}=====averSSIM:{:.4f}=====averSAM:{:.4f}====".format(np.mean(PSNRs), np.mean(SSIMs),
                                                                               np.mean(SAMs)))

if __name__ == "__main__":
    main()
