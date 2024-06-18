
import torch
from utils import psnr_loss, ssim, sam

eps = torch.finfo(torch.float32).eps

def get_metrics_reduced(img1, img2):
    # input: img1 {the pan-sharpened image}, img2 {the ground-truth image}
    # return: (larger better) psnr, ssim, scc, (smaller better) sam, ergas
    m1 = psnr_loss(img1, img2, 1.)
    m2 = ssim(img1, img2, 11, 'mean', 1.)
    m3 = cc(img1, img2)
    m4 = sam(img1, img2)
    m5 = ergas(img1, img2)
    return [m1.item(), m2.item(), m3.item(), m4.item(), m5.item()]

def fin_metrics_reduced(img1, img2, img3, img4):  # pre gt ms pan
    # input: img1 {the pan-sharpened image}, img2 {the ground-truth image}
    # return: (larger better) psnr, ssim, scc, (smaller better) sam, ergas
    np.seterr(divide='ignore', invalid='ignore')

    m1 = psnr_loss(img1, img2, 1.)
    m2 = ssim(img1, img2, 11, 'mean', 1.)
    m3 = cc(img1, img2)
    m4 = sam(img1, img2)
    m5 = ergas(img1, img2)

    img1 = img1.cpu().numpy()
    img2 = img2.cpu().numpy()
    img3 = img3.cpu().numpy()
    img4 = img4.cpu().numpy()

    m6 = qindex(img1, img2)
    m7 = D_lambda(img1, img3)
    m8 = D_s(img1, img3, img4)
    m9 = qnr(img1, img3, img4)

    return [m1.item(), m2.item(), m3.item(), m4.item(), m5.item(), m6.item(), m7.item(), m8.item(), m9.item()]

def ergas(img_fake, img_real, scale=4):
    """ERGAS for (N, C, H, W) image; torch.float32 [0.,1.].
    scale = spatial resolution of PAN / spatial resolution of MUL, default 4."""
    
    N,C,H,W = img_real.shape
    means_real = img_real.reshape(N,C,-1).mean(dim=-1)
    mses = ((img_fake - img_real)**2).reshape(N,C,-1).mean(dim=-1)
    # Warning: There is a small value in the denominator for numerical stability.
    # Since the default dtype of torch is float32, our result may be slightly different from matlab or numpy based ERGAS
    
    return 100 / scale * torch.sqrt((mses / (means_real**2 + eps)).mean())
    
def cc(img1, img2):
    """Correlation coefficient for (N, C, H, W) image; torch.float32 [0.,1.]."""
    N,C,_,_ = img1.shape
    img1 = img1.reshape(N,C,-1)
    img2 = img2.reshape(N,C,-1)
    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)
    cc = torch.sum(img1 * img2, dim=-1) / ( eps + torch.sqrt(torch.sum(img1**2, dim=-1)) * torch.sqrt(torch.sum(img2**2, dim=-1)) )
    cc = torch.clamp(cc, -1., 1.)
    return cc.mean(dim=-1)


# -*- coding: utf-8 -*-
"""
License: GNU-3.0
Code Reference:https://github.com/wasaCheney/IQA_pansharpening_python
"""

import numpy as np
from scipy import ndimage
import cv2


def _qindex(img1, img2, block_size=8):
    """Q-index for 2D (one-band) image, shape (H, W); uint or float [0, 1]"""
    # assert block_size > 1, 'block_size shold be greater than 1!'
    img1 = np.squeeze(img1)
    img2 = np.squeeze(img2)
    img1_ = img1.astype(np.float64)
    img2_ = img2.astype(np.float64)
    window = np.ones((block_size, block_size)) / (block_size ** 2)
    # window_size = block_size**2
    # filter, valid
    pad_topleft = int(np.floor(block_size / 2))
    pad_bottomright = block_size - 1 - pad_topleft
    mu1 = cv2.filter2D(img1_, -1, window)[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright]
    mu2 = cv2.filter2D(img2_, -1, window)[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(img1_ ** 2, -1, window)[pad_topleft:-pad_bottomright,
                pad_topleft:-pad_bottomright] - mu1_sq
    sigma2_sq = cv2.filter2D(img2_ ** 2, -1, window)[pad_topleft:-pad_bottomright,
                pad_topleft:-pad_bottomright] - mu2_sq
    #    print(mu1_mu2.shape)
    # print(sigma2_sq.shape)
    sigma12 = cv2.filter2D(img1_ * img2_, -1, window)[pad_topleft:-pad_bottomright,
              pad_topleft:-pad_bottomright] - mu1_mu2

    # all = 1, include the case of simga == mu == 0
    qindex_map = np.ones(sigma12.shape)
    # sigma == 0 and mu != 0

    #    print(np.min(sigma1_sq + sigma2_sq), np.min(mu1_sq + mu2_sq))

    idx = ((sigma1_sq + sigma2_sq) < 1e-8) * ((mu1_sq + mu2_sq) > 1e-8)
    qindex_map[idx] = 2 * mu1_mu2[idx] / (mu1_sq + mu2_sq)[idx]
    # sigma !=0 and mu == 0
    idx = ((sigma1_sq + sigma2_sq) > 1e-8) * ((mu1_sq + mu2_sq) < 1e-8)
    qindex_map[idx] = 2 * sigma12[idx] / (sigma1_sq + sigma2_sq)[idx]
    # sigma != 0 and mu != 0
    idx = ((sigma1_sq + sigma2_sq) > 1e-8) * ((mu1_sq + mu2_sq) > 1e-8)
    qindex_map[idx] = ((2 * mu1_mu2[idx]) * (2 * sigma12[idx])) / (
            (mu1_sq + mu2_sq)[idx] * (sigma1_sq + sigma2_sq)[idx])

    #    print(np.mean(qindex_map))

    #    idx = ((sigma1_sq + sigma2_sq) == 0) * ((mu1_sq + mu2_sq) != 0)
    #    qindex_map[idx] = 2 * mu1_mu2[idx] / (mu1_sq + mu2_sq)[idx]
    #    # sigma !=0 and mu == 0
    #    idx = ((sigma1_sq + sigma2_sq) != 0) * ((mu1_sq + mu2_sq) == 0)
    #    qindex_map[idx] = 2 * sigma12[idx] / (sigma1_sq + sigma2_sq)[idx]
    #    # sigma != 0 and mu != 0
    #    idx = ((sigma1_sq + sigma2_sq) != 0) * ((mu1_sq + mu2_sq) != 0)
    #    qindex_map[idx] =((2 * mu1_mu2[idx]) * (2 * sigma12[idx])) / (
    #        (mu1_sq + mu2_sq)[idx] * (sigma1_sq + sigma2_sq)[idx])

    return np.mean(qindex_map)


def qindex(img1, img2, block_size=8):
    """Q-index for 2D (H, W) or 3D (H, W, C) image; uint or float [0, 1]"""
    assert img1.shape[0]==1, 'Must be 1'

    qindexs = [_qindex(img1[:,i,:,:], img2[:,i,:,:], block_size) for i in range(img1.shape[1])]
    return np.array(qindexs).mean()


####################
# observation model
####################


def gaussian2d(N, std):
    t = np.arange(-(N - 1) // 2, (N + 2) // 2)
    t1, t2 = np.meshgrid(t, t)
    std = np.double(std)
    w = np.exp(-0.5 * (t1 / std) ** 2) * np.exp(-0.5 * (t2 / std) ** 2)
    return w
def kaiser2d(N, beta):
    t = np.arange(-(N - 1) // 2, (N + 2) // 2) / np.double(N - 1)
    t1, t2 = np.meshgrid(t, t)
    t12 = np.sqrt(t1 * t1 + t2 * t2)
    w1 = np.kaiser(N, beta)
    w = np.interp(t12, t, w1)
    w[t12 > t[-1]] = 0
    w[t12 < t[0]] = 0
    return w
def fir_filter_wind(Hd, w):
    """
    compute fir (finite impulse response) filter with window method
    Hd: desired freqeuncy response (2D)
    w: window (2D)
    """
    hd = np.rot90(np.fft.fftshift(np.rot90(Hd, 2)), 2)
    h = np.fft.fftshift(np.fft.ifft2(hd))
    h = np.rot90(h, 2)
    h = h * w
    h = h / np.sum(h)
    return h
def GNyq2win(GNyq, scale=4, N=41):
    """Generate a 2D convolutional window from a given GNyq
    GNyq: Nyquist frequency
    scale: spatial size of PAN / spatial size of MS
    """
    # fir filter with window method
    fcut = 1 / scale
    alpha = np.sqrt(((N - 1) * (fcut / 2)) ** 2 / (-2 * np.log(GNyq)))
    H = gaussian2d(N, alpha)
    Hd = H / np.max(H)
    w = kaiser2d(N, 0.5)
    h = fir_filter_wind(Hd, w)
    return np.real(h)
def mtf_resize(img, satellite='QuickBird', scale=4):
    # satellite GNyq
    scale = int(scale)
    if satellite == 'QuickBird':
        GNyq = [0.34, 0.32, 0.30, 0.22]  # Band Order: B,G,R,NIR
        GNyqPan = 0.15
    elif satellite == 'IKONOS':
        GNyq = [0.26, 0.28, 0.29, 0.28]  # Band Order: B,G,R,NIR
        GNyqPan = 0.17
    else:
        raise NotImplementedError('satellite: QuickBird or IKONOS')
    # lowpass
    img_ = img.squeeze()
    img_ = img_.astype(np.float64)
    if img_.ndim == 2:  # Pan
        H, W = img_.shape
        lowpass = GNyq2win(GNyqPan, scale, N=41)
    elif img_.ndim == 3:  # MS
        H, W, _ = img.shape
        lowpass = [GNyq2win(gnyq, scale, N=41) for gnyq in GNyq]
        lowpass = np.stack(lowpass, axis=-1)
    img_ = ndimage.filters.correlate(img_, lowpass, mode='nearest')
    # downsampling
    output_size = (H // scale, W // scale)
    img_ = cv2.resize(img_, dsize=output_size, interpolation=cv2.INTER_NEAREST)
    return img_

##################
# No reference IQA
##################

def D_lambda(img_fake, img_lm, block_size=32, p=1):
    """Spectral distortion
    img_fake, generated HRMS
    img_lm, LRMS"""
    # assert img_fake.ndim == img_lm.ndim == 3, 'Images must be 3D!'
    _, C_f, H_f, W_f = img_fake.shape
    _, C_r, H_r, W_r = img_lm.shape
    assert C_f == C_r, 'Fake and lm should have the same number of bands!'
    # D_lambda
    Q_fake = []
    Q_lm = []
    for i in range(C_f):
        for j in range(i + 1, C_f):
            # for fake
            band1 = img_fake[:,i,:,:]
            band2 = img_fake[:,j,:,:]
            Q_fake.append(_qindex(band1, band2, block_size=block_size))
            # for real
            band1 = img_lm[:,i,:,:]
            band2 = img_lm[:,j,:,:]
            Q_lm.append(_qindex(band1, band2, block_size=block_size))
    Q_fake = np.array(Q_fake)
    Q_lm = np.array(Q_lm)
    D_lambda_index = (np.abs(Q_fake - Q_lm) ** p).mean()
    return D_lambda_index ** (1 / p)


def D_s(img_fake, img_lm, pan, satellite='QuickBird', scale=4, block_size=32, q=1):
    """Spatial distortion
    img_fake, generated HRMS
    img_lm, LRMS
    pan, HRPan"""
    # fake and lm
    # assert img_fake.ndim == img_lm.ndim == 3, 'MS images must be 3D!'
    _, C_f, H_f, W_f = img_fake.shape
    _, C_r, H_r, W_r = img_lm.shape
    assert H_f // H_r == W_f // W_r == scale, 'Spatial resolution should be compatible with scale'
    assert C_f == C_r, 'Fake and lm should have the same number of bands!'
    # fake and pan
    # assert pan.ndim == 3, 'Panchromatic image must be 3D!'
    _, C_p, H_p, W_p = pan.shape
    assert C_p == 1, 'size of 3rd dim of Panchromatic image must be 1'
    assert H_f == H_p and W_f == W_p, "Pan's and fake's spatial resolution should be the same"
    # get LRPan, 2D
    pan_lr = mtf_resize(pan, satellite=satellite, scale=scale)
    # print(pan_lr.shape)
    # D_s
    Q_hr = []
    Q_lr = []
    for i in range(C_f):
        # for HR fake
        band1 = img_fake[:,i,:,:]
        band2 = pan[:,0,:,:]  # the input PAN is 3D with size=1 along 3rd dim
        # print(band1.shape)
        # print(band2.shape)
        Q_hr.append(_qindex(band1, band2, block_size=block_size))
        band1 = img_lm[:,i,:,:]
        band2 = pan_lr  # this is 2D
        # print(band1.shape)
        # print(band2.shape)
        Q_lr.append(_qindex(band1, band2, block_size=block_size))
    Q_hr = np.array(Q_hr)
    Q_lr = np.array(Q_lr)
    D_s_index = (np.abs(Q_hr - Q_lr) ** q).mean()
    return D_s_index ** (1 / q)

def qnr(img_fake, img_lm, pan, satellite='QuickBird', scale=4, block_size=32, p=1, q=1, alpha=1, beta=1):
    """QNR - No reference IQA"""
    D_lambda_idx = D_lambda(img_fake, img_lm, block_size, p)
    D_s_idx = D_s(img_fake, img_lm, pan, satellite, scale, block_size, q)
    QNR_idx = (1 - D_lambda_idx) ** alpha * (1 - D_s_idx) ** beta
    # print(img_fake)
    return QNR_idx


def no_ref_evaluate(pred, pan, hs):
    # no reference metrics
    c_D_lambda = D_lambda(pred, hs)
    c_D_s = D_s(pred, hs, pan)
    c_qnr = qnr(pred, hs, pan)
    return [c_D_lambda, c_D_s, c_qnr]





