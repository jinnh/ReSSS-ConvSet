import os, sys
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import numpy as np
import torch.optim as optim
from os import path as osp
from torch.utils.data import DataLoader
from Hyper_loader import Hyper_dataset
from option import opt
from utils.metrics import compare_mpsnr, compare_mssim, cal_sam
from torch.optim.lr_scheduler import MultiStepLR
from utils.serialization import Logger

if opt.model_name == 'res3conv':
    from model.res3conv import reconnetHRHSI as HSIG
else:
    from model.res3net import reconnetHRHSI as HSIG

save_dir = './checkpoints/'+ opt.dataset +'_x'+ str(opt.upscale_factor)

def main():
    if opt.cuda:
        print("=> Use GPU ID: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    torch.manual_seed(opt.seed)

    if opt.model_name == 'res3conv':
        sys.stdout = Logger(osp.join(save_dir, 'log_train_res3conv.txt'))
    else:
        sys.stdout = Logger(osp.join(save_dir, 'log_train_res3net.txt'))

    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)
    cudnn.benchmark = True

    # Loading datasets
    data_name = opt.dataset
    Hyper_test = Hyper_dataset(output_shape=128, Training_mode='Test', data_name=data_name)
    val_loader = DataLoader(Hyper_test, batch_size=1, shuffle=False, num_workers=1, pin_memory=True,
                            drop_last=True)

    Hyper_train = Hyper_dataset(output_shape=128, Training_mode='Train', data_name=data_name)
    train_loader = DataLoader(Hyper_train, batch_size=opt.batchSize, shuffle=True, num_workers=4, pin_memory=True,
                             drop_last=True)

    # Buliding model
    model = HSIG()
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
            
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-08)

    # resuming from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            param_dict = torch.load(opt.resume)
            opt.start_epoch = 1
            for i in param_dict['model']:
               model.state_dict()[i].copy_(param_dict['model'][i])
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # Setting scheduler
    scheduler = MultiStepLR(optimizer, milestones=[25, 50], gamma=0.5, last_epoch=-1)

    # Training
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        torch.cuda.empty_cache()
        scheduler.step()
        train(train_loader, optimizer, model, criterion, epoch)
        if (epoch % 50 == 0):
            val(val_loader, model)
            save_checkpoint(epoch, model, optimizer)

def train(train_loader, optimizer, model, criterion, epoch):
    torch.cuda.empty_cache()
    print("Epoch = {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"]))
    model.train()
    lossValue = []

    for iteration, (input, label) in enumerate(train_loader, 1):
        if opt.cuda:
            input = input.cuda()
            label = label.cuda()
        if opt.model_name == 'res3conv':
            SR = model(input)
            svdloss = 0
        else:
            SR, svdloss = model(input)
            svdloss = svdloss * opt.svdLossWeight
        l1_loss = criterion(SR, label)
        loss = l1_loss - svdloss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lossValue.append(loss.item())

        if iteration % 80 == 0:
            print("===> {} Epoch[{}]({}/{}): Loss: {:.7f} L1Loss: {:.7f} svdLoss: {:.7f}".format(time.ctime(), epoch,
                                                                                                 iteration,
                                                                                                 len(train_loader),
                                                                                                 loss.item(),
                                                                                                 l1_loss.item(),
                                                                                                 svdloss.item()))

    print("===> {} Epoch[{}]: AveLoss: {:.10f}".format(time.ctime(), epoch, np.mean(np.array(lossValue))))


def val(val_loader, model):
    torch.cuda.empty_cache()
    with torch.no_grad():
        model.eval()
        val_psnr = 0
        val_ssim = 0
        val_sam = 0

        for iteration, (input, HR) in enumerate(val_loader, 1):

            if opt.cuda:
                input = input.cuda()
                HR = HR.cuda()

            if opt.model_name == 'res3conv':
                SR = model(input)
            else:
                SR, _ = model(input)
            SR = SR.cpu().data[0].numpy()
            HR = HR.cpu().data[0].numpy()
            val_psnr += compare_mpsnr(x_true=HR, x_pred=SR, data_range=1)
            val_ssim += compare_mssim(x_true=HR, x_pred=SR, data_range=1, multidimension=False)
            val_sam += cal_sam(SR, HR)
        val_psnr = val_psnr / len(val_loader)
        val_ssim = val_ssim / len(val_loader)
        val_sam = val_sam / len(val_loader)
        print("===> PSNR={:.4f}, SSIM={:.4f}, SAM={:.4f} ".format(val_psnr,val_ssim,val_sam))


def save_checkpoint(epoch, model, optimizer):
    model_out_path = save_dir + "/{}_{}_epoch_{}.pth".format(opt.model_name, opt.upscale_factor, epoch)
    state = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()}
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(state, model_out_path)


if __name__ == "__main__":
    main()

