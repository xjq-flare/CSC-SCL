if __name__ == '__main__':
    import sys
    sys.path.insert(0, '../')


try:
    import calc_uciqe
    import calc_niqe
    import matlab
except Exception as e:
    print(e)
    pass

import os
import cv2
from statistics import mean
import numpy as np
from torchvision import transforms
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from utils.no_reference_evaluation import getUCIQE
from utils.uiqm_utils import getUIQM
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio


def identify(ob):
    return ob


def calc_ref_img(gt_image_path, pred_image_path, size=None):
    if size is None:
        transform = identify
    else:
        transform = transforms.Resize(size=size)

    y_true = np.array(transform(Image.open(gt_image_path)))
    y_pred = np.array(transform(Image.open(pred_image_path)))

    psnr1 = psnr(y_true[:, :, 0], y_pred[:, :, 0], data_range=255)
    psnr2 = psnr(y_true[:, :, 1], y_pred[:, :, 1], data_range=255)
    psnr3 = psnr(y_true[:, :, 2], y_pred[:, :, 2], data_range=255)
    psnr_value = (psnr1 + psnr2 + psnr3) / 3.0
    ssim_value = ssim(y_true, y_pred, channel_axis=-1)

    return float(psnr_value), float(ssim_value)


def calc_uciqe_img(pred_image_path, size=None):
    image = cv2.imread(pred_image_path)
    if size is not None:
        image = cv2.resize(image, size)
    uciqe = getUCIQE(image)

    return float(uciqe)


def calc_uiqm_img(pred_image_path, size=None):
    image = cv2.imread(pred_image_path)
    if size is not None:
        image = cv2.resize(image, size)
    uiqm, uicm, uism, uiconm = getUIQM(image)

    return float(uiqm), float(uicm), float(uism), float(uiconm)


def calc_ref(true_folder, pred_folder, size=None):
    if size is None:
        transform = identify
    else:
        transform = transforms.Resize(size=size)

    path_list = os.listdir(true_folder)
    PSNR = []
    SSIM = []

    for item in path_list:
        try:
            y_true = np.array(transform(Image.open(os.path.join(true_folder, item))))
            y_pred = np.array(transform(Image.open(os.path.join(pred_folder, item))))
        except (FileNotFoundError, IsADirectoryError) as e:
            print(f"File not found: item: {item} {e}")
            continue

        psnr1 = psnr(y_true[:, :, 0], y_pred[:, :, 0])
        psnr2 = psnr(y_true[:, :, 1], y_pred[:, :, 1])
        psnr3 = psnr(y_true[:, :, 2], y_pred[:, :, 2])
        psnr_value = (psnr1 + psnr2 + psnr3) / 3.0
        PSNR.append(psnr_value)
        ssim_value = ssim(y_true, y_pred, channel_axis=-1)
        SSIM.append(ssim_value)
    return mean(PSNR), mean(SSIM)


def calc_all(pred_folder, true_folder, size_ref=None, size_no_ref=None):
    PSNR, SSIM = [], []
    UIQM, UICM, UISM, UICONM, UCIQE = [], [], [], [], []

    # calculate psnr, ssim
    if size_ref is None:
        transform = identify
    else:
        transform = transforms.Resize(size=size_ref)

    for item in os.listdir(true_folder):
        y_pred_path = os.path.join(pred_folder, item)
        try:
            y_true = np.array(transform(Image.open(os.path.join(true_folder, item))))
            y_pred = np.array(transform(Image.open(y_pred_path)))
        except (FileNotFoundError, IsADirectoryError) as e:
            print(f"File not found: item: {item} {e}")
            continue

        psnr1 = psnr(y_true[:, :, 0], y_pred[:, :, 0])
        psnr2 = psnr(y_true[:, :, 1], y_pred[:, :, 1])
        psnr3 = psnr(y_true[:, :, 2], y_pred[:, :, 2])
        psnr_value = (psnr1 + psnr2 + psnr3) / 3.0
        PSNR.append(psnr_value)

        ssim_value = ssim(y_true, y_pred, channel_axis=-1)
        SSIM.append(ssim_value)

        # calculate UIQM
        image_pred = cv2.imread(y_pred_path)
        if size_no_ref is not None:
            image_pred = cv2.resize(image_pred, size_no_ref)
        uiqm, uicm, uism, uiconm = getUIQM(image_pred)
        UIQM.append(uiqm)
        UICM.append(uicm)
        UISM.append(uism)
        UICONM.append(uiconm)

        # calculate UCIQE
        uciqe = getUCIQE(image_pred)
        UCIQE.append(uciqe)

    return mean(PSNR), mean(SSIM), mean(UIQM), mean(UICM), mean(UISM), mean(UICONM), mean(UCIQE)


def calc_all_metrics(gen_folder, gt_folder, size_ref=(256, 256), size_no_ref=None):
    PSNR, SSIM = [], []
    UIQM, UICM, UISM, UICONM, UCIQE = [], [], [], [], []
    NIQE = []
    my_calc_uciqe = calc_uciqe.initialize()
    my_calc_niqe = calc_niqe.initialize()

    transform = transforms.Compose([transforms.Resize(size_ref),
                                    transforms.ToTensor()])  
    calc_ssim = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
    calc_psnr = PeakSignalNoiseRatio().cuda()

    for img in os.listdir(gt_folder):
        if os.path.isdir(img):
            continue
        gen_img_path = os.path.join(gen_folder, img)
        gt_img_path = os.path.join(gt_folder, img)
        gen_img = Image.open(gen_img_path)
        gt_img = Image.open(gt_img_path)

        gen_img_tensor = transform(gen_img).unsqueeze(0).cuda()
        gt_img_tensor = transform(gt_img).unsqueeze(0).cuda()
        SSIM.append(float(calc_ssim(gen_img_tensor, gt_img_tensor)))
        PSNR.append(float(calc_psnr(gen_img_tensor, gt_img_tensor)))
        uiqm_value, uicm_value, uism_value, uiconm_value = calc_uiqm_img(gen_img_path, size_no_ref)
        if size_no_ref is not None:
            uciqe_value = my_calc_uciqe.calc_uciqe(os.path.abspath(gen_img_path), matlab.int32(list(size_no_ref)))
            niqe_value = my_calc_niqe.calc_niqe(os.path.abspath(gen_img_path), matlab.int32(list(size_no_ref)))
        else:
            uciqe_value = my_calc_uciqe.calc_uciqe(os.path.abspath(gen_img_path))
            niqe_value = my_calc_niqe.calc_niqe(os.path.abspath(gen_img_path))

        UIQM.append(uiqm_value)
        UICM.append(uicm_value)
        UISM.append(uism_value)
        UICONM.append(uiconm_value)
        UCIQE.append(uciqe_value)
        NIQE.append(niqe_value)

    my_calc_uciqe.terminate()
    my_calc_niqe.terminate()

    return mean(PSNR), mean(SSIM), mean(UIQM), mean(UICM), mean(UISM), mean(UICONM), mean(UCIQE), mean(NIQE)


def calc_psrn(true_folder, pred_folder, size=None):
    if size is None:
        transform = identify
    else:
        transform = transforms.Resize(size=size)

    path_list = os.listdir(true_folder)
    PSNR = []

    for item in path_list:
        y_true = np.array(transform(Image.open(os.path.join(true_folder, item))))
        y_pred = np.array(transform(Image.open(os.path.join(pred_folder, item))))

        psnr1 = psnr(y_true[:, :, 0], y_pred[:, :, 0])
        psnr2 = psnr(y_true[:, :, 1], y_pred[:, :, 1])
        psnr3 = psnr(y_true[:, :, 2], y_pred[:, :, 2])
        psnr_value = (psnr1 + psnr2 + psnr3) / 3.0
        PSNR.append(psnr_value)

    return np.array(PSNR).mean()


def calc_ssim(true_folder, pred_folder, size=None):
    if size is None:
        transform = identify
    else:
        transform = transforms.Resize(size=size)

    path_list = os.listdir(true_folder)
    SSIM = []

    for item in path_list:
        y_true = np.array(transform(Image.open(os.path.join(true_folder, item))))
        y_pred = np.array(transform(Image.open(os.path.join(pred_folder, item))))

        ssim_value = ssim(y_true, y_pred, channel_axis=-1)
        SSIM.append(ssim_value)

    return np.array(SSIM).mean()


def calc_UIQM(pred_image_folder, size=None):
    uiqms = []
    uicms = []
    uisms = []
    uiconms = []
    for img in os.listdir(pred_image_folder):
        image = os.path.join(pred_image_folder, img)

        image = cv2.imread(image)
        if size is not None:
            image = cv2.resize(image, size)

        # calculate UIQM
        uiqm, uicm, uism, uiconm = getUIQM(image)
        uiqms.append(uiqm)
        uicms.append(uicm)
        uisms.append(uism)
        uiconms.append(uiconm)

    return np.array(uiqms), np.array(uicms), np.array(uisms), np.array(uiconms)


if __name__ == '__main__':
    import sys

    sys.path.insert(0, '../')

    pred_folder = r"D:\images\valA"
    true_folder = r"D:\images\valB_gt"

    # print(calc_ref(true_folder, pred_folder, size=(256, 256)))
    print(calc_all(true_folder, pred_folder, (256, 256)))
