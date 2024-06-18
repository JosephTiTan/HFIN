# -*- coding: utf-8 -*-

'''
------------------------------------------------------------------------------
Import packages
------------------------------------------------------------------------------
'''

import os
# import xlwt
import time
import datetime
import numpy as np

import torch
import torch.nn as nn

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from scipy.io import savemat
import cv2
import sys

sys.path.append("E:/Project/Pansharpening/clswin")

from models.HFIN import Net

from metrics import get_metrics_reduced
from metrics import fin_metrics_reduced
from utils import PSH5Datasetfu, PSDataset, prepare_data, normlization, save_param, psnr_loss, ssim, save_img
from data import Data


'''
------------------------------------------------------------------------------
Configure our network
------------------------------------------------------------------------------
'''

model_str = 'HFIN'
satellite_str = 'WV2'   # WV2 WV3 GF2


alfa = 0.1
ms_channels = 4
pan_channels = 1
scale = 4

# . Set the hyper-parameters for training
num_epochs = 1000
lr = 1e-3
print('lr is', lr)
weight_decay = 0
batch_size = 4
n_layer = 8
n_feat = 16

# . Get your model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.set_device(0)
net = Net(4).to(device)  #channels=4 for gppnn
print(net)

# . Get your optimizer, scheduler and loss function
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
loss_fn = nn.L1Loss().to(device)

# . Create your data loaders
data_dir_ms_train = 'E:/Project/Pansharpening/yaogan/WV2_data/train128/ms/'
data_dir_pan_train = 'E:/Project/Pansharpening/yaogan/WV2_data/train128/pan/'

trainloader = DataLoader(Data(data_dir_ms=data_dir_ms_train, data_dir_pan=data_dir_pan_train),
                         batch_size=batch_size,
                         shuffle=True)

data_dir_ms_test = 'E:/Project/Pansharpening/yaogan/WV2_data/test128/ms/'
data_dir_pan_test = 'E:/Project/Pansharpening/yaogan/WV2_data/test128/pan/'
testloader = DataLoader(Data(data_dir_ms=data_dir_ms_test, data_dir_pan=data_dir_pan_test),
                        batch_size=1)  # [N,C,K,H,W]

loader = {'train': trainloader,
          'validation': testloader}

# . Creat logger
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
save_path = os.path.join(
    'E:/Project/Pansharpening/clswin/training/logs/%s' % (model_str),
    timestamp + '_%s_layer%d_filter_%d' % (satellite_str, n_layer, n_feat)
)
writer = SummaryWriter(save_path)
params = {'model': model_str,
          'satellite': satellite_str,
          'epoch': num_epochs,
          'lr': lr,
          'batch_size': batch_size,
          'n_feat': n_feat,
          'n_layer': n_layer}
save_param(params,
           os.path.join(save_path, 'param.json'))

'''
------------------------------------------------------------------------------
Train
------------------------------------------------------------------------------
'''

step = 0
best_psnr_val, psnr_val, ssim_val = 0., 0., 0.
torch.backends.cudnn.benchmark = True
prev_time = time.time()

for epoch in range(num_epochs):
    ''' train '''
    for i, (ms, pan, gt) in enumerate(loader['train']):
        # 0. preprocess data
        ms, pan, gt = ms.cuda(), pan.cuda(), gt.cuda()
        # 1. update
        net.train()
        net.zero_grad()
        optimizer.zero_grad()
        #[4, 4, 32, 32]) torch.Size([4, 1, 128, 128])
        predHR= net(ms, pan)
        loss_forward = loss_fn(predHR, gt.detach())
        loss = loss_forward
        loss.backward()
        optimizer.step()

        # 2. print
        # Determine approximate time left
        batches_done = epoch * len(loader['train']) + i
        batches_left = num_epochs * len(loader['train']) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [loss: %f] [forward loss: %f] [PSNR/Best: %.4f/%.4f] ETA: %s"
            % (
                epoch,
                num_epochs,
                i,
                len(loader['train']),
                loss.item(),
                loss_forward.item(),
                # loss_backward.item(),
                psnr_val,
                best_psnr_val,
                time_left,
            )
        )

        # 3. Log the scalar values
        writer.add_scalar('loss', loss.item(), step)
        writer.add_scalar('learning rate', optimizer.state_dict()['param_groups'][0]['lr'], step)
        step += 1

    ''' validation '''
    current_psnr_val = psnr_val

    psnr_val = 0.
    ssim_val = 0.
    metrics = torch.zeros(2, testloader.__len__())
    with torch.no_grad():
        net.eval()
        for i, (ms, pan, gt) in enumerate(testloader):
            ms, pan, gt = ms.cuda(), pan.cuda(), gt.cuda()

            predHR = net(ms, pan)
            metrics[:, i] = torch.Tensor(get_metrics_reduced(predHR, gt))[:2]
        psnr_val, ssim_val = metrics.mean(dim=1)
    writer.add_scalar('PSNR/test', psnr_val, epoch)
    writer.add_scalar('SSIM/test', ssim_val, epoch)

    ''' save model '''
    # Save the best weight
    if best_psnr_val < psnr_val:
        best_psnr_val = psnr_val
        torch.save({'net': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch},
                   os.path.join(save_path, 'best_net.pth'))
    # Save the current weight
    torch.save({'net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch},
               os.path.join(save_path, 'last_net.pth'))

    ''' backtracking '''
    if epoch > 0:
        if torch.isnan(loss):
            print(10 * '=' + 'Backtracking!' + 10 * '=')
            net.load_state_dict(torch.load(os.path.join(save_path, 'best_net.pth'))['net'])
            optimizer.load_state_dict(torch.load(os.path.join(save_path, 'best_net.pth'))['optimizer'])

'''
------------------------------------------------------------------------------
Test
------------------------------------------------------------------------------
'''

# 1. Load the best weight and create the dataloader for testing
net.load_state_dict(torch.load(os.path.join( save_path, 'best_net.pth'))['net'])

# 2. Compute the metrics
metrics = torch.zeros(9, testloader.__len__())
with torch.no_grad():
    net.eval()
    for i, (ms, pan, gt) in enumerate(testloader):
        ms, pan, gt = ms.cuda(), pan.cuda(), gt.cuda()
        # ms,_ = normlization(ms.cuda())
        # pan,_ = normlization(pan.cuda())
        # gt,_ = normlization(gt.cuda())
        predHR = net(ms, pan)
        metrics[:, i] = torch.Tensor(fin_metrics_reduced(predHR, gt, ms, pan))
        # savemat(os.path.join(save_path, str(i)),
        #         {'HR': imgf.squeeze().detach().cpu().numpy()})
        save_img(save_path, str(i)+'.tiff',predHR)

list_PSNR = []
list_SSIM = []
list_CC = []
list_SAM = []
list_ERGAS = []

list_Q = []
list_DL = []
list_DS = []
list_QNR = []
for n in range(testloader.__len__()):
    list_PSNR.append(metrics[0, n])
    list_SSIM.append(metrics[1, n])
    list_CC.append(metrics[2, n])
    list_SAM.append(metrics[3, n])
    list_ERGAS.append(metrics[4, n])

    list_Q.append(metrics[5, n])
    list_DL.append(metrics[6, n])
    list_DS.append(metrics[7, n])
    list_QNR.append(metrics[8, n])

print("list_psnr_mean:", np.mean(list_PSNR))
print("list_ssim_mean:", np.mean(list_SSIM))
print("list_cc_mean:", np.mean(list_CC))
print("list_sam_mean:", np.mean(list_SAM))
print("list_ergas_mean:", np.mean(list_ERGAS))

print("list_qindex_mean:", np.mean(list_Q))
print("list_dlama_mean:", np.mean(list_DL))
print("list_ds_mean:", np.mean(list_DS))
print("list_qnr_mean:", np.mean(list_QNR))
