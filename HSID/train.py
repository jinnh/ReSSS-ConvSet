import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import numpy as np
import torch.optim as optim
from os import path as osp
from torch.utils.data import DataLoader
from Hyper_loader import Hyper_dataset
from metrics import compare_mpsnr, compare_mssim, cal_sam
from option import opt
from torch.optim.lr_scheduler import MultiStepLR
from serialization import Logger
import os, sys
import time

if opt.model_name == 'res3conv':
    from model.res3conv import HSIDframework as HSID
else:
    from model.res3net import HSIDframework as HSID

save_dir = './checkpoints/'+ opt.dataset + '/'

def main():
    if opt.cuda:
        print("=> Use GPU ID: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    torch.manual_seed(opt.seed)
    if opt.model_name == 'res3conv':
        sys.stdout = Logger(osp.join(save_dir, 'log_train_gaussian_res3conv.txt'))
    else:
        sys.stdout = Logger(osp.join(save_dir, 'log_train_gaussian_res3net.txt'))

    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)
    cudnn.benchmark = True

    data_name = opt.dataset
    Hyper_train = Hyper_dataset(Training_mode='Train', data_name=data_name)
    train_loader = DataLoader(Hyper_train, batch_size=opt.batchSize, shuffle=True, num_workers=4, pin_memory=True,
                             drop_last=True)

    Hyper_test = Hyper_dataset(Training_mode='Val', data_name=data_name)
    val_loader = DataLoader(Hyper_test, batch_size=1, shuffle=False, num_workers=1, pin_memory=True,
                            drop_last=True)

    # Buliding model
    model = HSID()
    criterion = nn.L1Loss()

    if opt.cuda:
        model = nn.DataParallel(model).cuda()
        criterion = criterion.cuda()
    else:
        model = model.cpu()
    print('# parameters:', sum(param.numel() for param in model.parameters()))

    for name, p in model.named_parameters():
        if 'downscale' in name:
            p.requires_grad = False

    # Setting Optimizer
    optimizer = optim.Adam(filter(lambda x: x.requires_grad is not False ,model.parameters()), lr=opt.lr, betas=(0.9, 0.999), eps=1e-08)

    # Resuming from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            param_dict = torch.load(opt.resume)
            opt.start_epoch = 1
            for i in param_dict['model']:
                model.state_dict()[i].copy_(param_dict['model'][i])
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # Setting learning rate
    scheduler = MultiStepLR(optimizer, milestones=[5, 10, 15, 20], gamma=0.5, last_epoch=-1)

    # Training
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):

        torch.cuda.empty_cache()
        scheduler.step()
        val(val_loader, model, epoch)
        train(train_loader, optimizer, model, criterion, epoch)
        if (epoch % 25 == 0 or epoch == opt.nEpochs):
            # val(val_loader, model, epoch)
            save_checkpoint(epoch, model, optimizer)



def train(train_loader, optimizer, model, criterion, epoch):
    torch.cuda.empty_cache()
    print("Epoch = {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"]))
    model.train()
    lossValue = []

    for iteration, (input, gt) in enumerate(train_loader, 1):
        if opt.cuda:
            input = input.cuda()
            gt = gt.cuda()

        if opt.model_name == 'res3conv':
            HSI_Denoised = model(input)
            l1_loss = criterion(HSI_Denoised, gt)
            loss = l1_loss
        else:
            HSI_Denoised, svd_loss = model(input, 'training')
            svd_loss = svd_loss * opt.svdLossWeight
            l1_loss = criterion(HSI_Denoised, gt)
            loss = l1_loss - svd_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lossValue.append(loss.item())

        if iteration % 1000 == 0:
            if opt.model_name == 'res3conv':
                print("===> {} Epoch[{}]({}/{}): Loss: {:.7f} L1Loss: {:.7f}".format(time.ctime(), epoch, iteration, len(train_loader), loss.item(), l1_loss.item()))
            else:
                print("===> {} Epoch[{}]({}/{}): Loss: {:.7f} L1Loss: {:.7f} svdLoss: {:.7f}".format(time.ctime(), epoch, iteration, len(train_loader), loss.item(), l1_loss.item(), svd_loss.item()))

    print("===> {} Epoch[{}]: AveLoss: {:.7f}".format(time.ctime(), epoch, np.mean(np.array(lossValue))))

def val(val_loader, model, epoch):

    torch.cuda.empty_cache()
    with torch.no_grad():
        model.eval()
        val_psnr = 0
        val_ssim = 0
        val_sam = 0

        for iteration, (input, GT) in enumerate(val_loader, 1):

            if opt.cuda:
                input = input.cuda()
                GT = GT.cuda()

            if opt.model_name == 'res3conv':
                HSI_Denoised = model(input)
            else:
                HSI_Denoised, _ = model(input, 'val')
            HSI_Denoised = HSI_Denoised.cpu().data[0].numpy()
            GT = GT.cpu().data[0].numpy()
            val_psnr += compare_mpsnr(x_true=GT, x_pred=HSI_Denoised, data_range=1)
            val_ssim += compare_mssim(x_true=GT, x_pred=HSI_Denoised, data_range=1, multidimension=False)
            val_sam += cal_sam(GT, HSI_Denoised)
        val_psnr = val_psnr / len(val_loader)
        val_ssim = val_ssim / len(val_loader)
        val_sam = val_sam / len(val_loader)
        print("===> {} Epoch[{}]: PSNR={:.4f}, SSIM={:.4f}, SAM={:.4f} ".format(time.ctime(), epoch, val_psnr, val_ssim, val_sam))

def save_checkpoint(epoch, model, optimizer):
    model_out_path = save_dir + '/' + "{}_{}_epoch_{}.pth".format(opt.model_name, opt.noiseType, epoch)
    state = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()}
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(state, model_out_path)


if __name__ == "__main__":
    main()

