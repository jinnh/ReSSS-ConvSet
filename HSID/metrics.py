import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.mat'])


def compare_rmse(x_true, x_pred):
    """
    Calculate Root mean squared error
    :param x_true:
    :param x_pred:
    :return:
    """
    x_true, x_pred = x_true.astype(np.float32), x_pred.astype(np.float32)
    return np.linalg.norm(x_true - x_pred) / (np.sqrt(x_true.shape[0] * x_true.shape[1] * x_true.shape[2]))


def compare_mpsnr(x_true, x_pred, data_range):
    """
    :param x_true: Input image must have three dimension (H, W, C)
    :param x_pred:
    :return:
    """
    x_true, x_pred = x_true.astype(np.float32), x_pred.astype(np.float32)
    channels = x_true.shape[0]
    # total_psnr = [peak_signal_noise_ratio(x_true[k, :, :], x_pred[k, :, :], data_range=data_range)
    #              for k in range(channels)]
    total_psnr = []
    for k in range(channels):
        psnr = peak_signal_noise_ratio(x_true[k, :, :], x_pred[k, :, :], data_range=data_range)
        # print(k, psnr)
        total_psnr.append(psnr)

    return np.mean(total_psnr)


def compare_mssim(x_true, x_pred, data_range, multidimension):
    """

    :param x_true:
    :param x_pred:
    :param data_range:
    :param multidimension:
    :return:
    """
    mssim = [
        structural_similarity(x_true[i, :, :], x_pred[i, :, :], data_range=data_range, multidimension=multidimension)
        for i in range(x_true.shape[0])]

    return np.mean(mssim)


def cal_sam(X, Y, eps=1e-8):
    tmp = (np.sum(X * Y, axis=0) + eps) / (np.sqrt(np.sum(X ** 2, axis=0)) + eps) / (
                np.sqrt(np.sum(Y ** 2, axis=0)) + eps)
    return np.mean(np.real(np.arccos(tmp)))