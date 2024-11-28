has_mr = True
try:
    import calc_uciqe
    import matlab
except Exception as e:
    print(e)
    RED = '\033[91m'
    RESET = '\033[0m'
    warning_message = (
        "UCIQE metric cannot be calculated because MATLAB Runtime is not installed or not correctly configured. "
        "Please install MATLAB Runtime by following the instructions at "
        "https://ww2.mathworks.cn/help/compiler/install-the-matlab-runtime.html"
    )
    print(RED + warning_message + RESET)
    has_mr = False

import os
import argparse
from PIL import Image
from statistics import mean
from utils.metrics import calc_uiqm_img
from torchvision import transforms
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio


def calc_metrics_with_gt(gen_folder, gt_folder):
    PSNR, SSIM = [], []
    UIQM, UICM, UISM, UICONM, UCIQE = [], [], [], [], []
    if has_mr:
        my_calc_uciqe = calc_uciqe.initialize()

    transform = transforms.Compose([transforms.Resize(size_ref),
                                    transforms.ToTensor()])
    calc_ssim = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
    calc_psnr = PeakSignalNoiseRatio().cuda()

    count = 0

    for img in os.listdir(gt_folder):
        if test:
            name, suffix = img.split('.')
            if single:
                gen_img_path = os.path.join(gen_folder, name + '_fake' + '.' + suffix)
            else:
                gen_img_path = os.path.join(gen_folder, name + '_fake_B' + '.' + suffix)
        else:
            gen_img_path = os.path.join(gen_folder, img)

        gt_img_path = os.path.join(gt_folder, img)
        gen_img = Image.open(gen_img_path)
        gt_img = Image.open(gt_img_path)
        gen_img_tensor = transform(gen_img).unsqueeze(0).cuda()
        gt_img_tensor = transform(gt_img).unsqueeze(0).cuda()
        SSIM.append(float(calc_ssim(gen_img_tensor, gt_img_tensor)))
        PSNR.append(float(calc_psnr(gen_img_tensor, gt_img_tensor)))

        uiqm_value, uicm_value, uism_value, uiconm_value = calc_uiqm_img(gen_img_path, size_no_ref)

        if has_mr:
            try:
                if size_no_ref is not None:
                    uciqe_value = my_calc_uciqe.calc_uciqe(os.path.abspath(gen_img_path), matlab.int32(list(size_no_ref)))
                else:
                    uciqe_value = my_calc_uciqe.calc_uciqe(os.path.abspath(gen_img_path))
            except ZeroDivisionError as e:
                print(e)
                print(f"img: {gen_img_path}")
                continue
        else:
            uciqe_value = 0

        count += 1

        UIQM.append(uiqm_value)
        UICM.append(uicm_value)
        UISM.append(uism_value)
        UICONM.append(uiconm_value)
        UCIQE.append(uciqe_value)

    psnr = mean(PSNR)
    ssim = mean(SSIM)
    uiqm = mean(UIQM)
    uicm = mean(UICM)
    uism = mean(UISM)
    uiconm = mean(UICONM)
    uciqe = mean(UCIQE)

    if has_mr:
        my_calc_uciqe.terminate()

    print(f"calc_metrics_with_gt: calc {count} images")
    with_gt_metrics = f"PSNR: {psnr:.4f}, SSIM:{ssim:.4f}, UCIQE: {uciqe:.4f}, UIQM: {uiqm:.4f}, UICM: {uicm:.4f}, UISM: {uism:.4f}, UICONM: {uiconm:.4f}"

    return with_gt_metrics


def calc_metrics_without_gt(gen_folder):
    UIQM, UICM, UISM, UICONM, UCIQE = [], [], [], [], []
    
    if has_mr:
        my_calc_uciqe = calc_uciqe.initialize()

    count = 0
    for img in os.listdir(gen_folder):
        name, _ = os.path.splitext(img)
        if not test or (test and name.endswith("fake")):
            gen_img_path = os.path.join(gen_folder, img)
            uiqm_value, uicm_value, uism_value, uiconm_value = calc_uiqm_img(gen_img_path, size_no_ref)

            if has_mr:
                try:
                    if size_no_ref is not None:
                        uciqe_value = my_calc_uciqe.calc_uciqe(os.path.abspath(gen_img_path),
                                                            matlab.int32(list(size_no_ref)))
                    else:
                        uciqe_value = my_calc_uciqe.calc_uciqe(os.path.abspath(gen_img_path))
                    count += 1
                except ZeroDivisionError as e:
                    print(e)
                    print(f"img: {gen_img_path}")
                    continue
            else:
                uciqe_value = 0

            UIQM.append(uiqm_value)
            UICM.append(uicm_value)
            UISM.append(uism_value)
            UICONM.append(uiconm_value)
            UCIQE.append(uciqe_value)

    if has_mr:
        my_calc_uciqe.terminate()

    print(f"calc_metrics_without_gt: calc {count} images")
    without_gt_metrics = f"UCIQE: {mean(UCIQE):.4f}, UIQM: {mean(UIQM):.4f}, UICM: {mean(UICM):.4f}, UISM: {mean(UISM):.4f}, UICONM: {mean(UICONM):.4f}"

    return without_gt_metrics


def print_write(gen_folder, gt_folder, metrics_str):
    print(metrics_str)
    result_file = open(os.path.join(os.path.split(gen_folder)[0], "cal_metrics.txt"), 'a', encoding='utf-8')
    result_file.write(f"calc ref: gen: {gen_folder} and gt: {gt_folder}\n")
    result_file.write(f"{metrics_str}\n\n")
    result_file.close()

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen', type=str, default='')
    parser.add_argument('--gt', type=str, default='')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--single', action='store_true') 
    parser.add_argument('--noref', action='store_true')
    opt = parser.parse_args()

    return opt


if __name__ == "__main__":
    size_ref = (256, 256)
    size_no_ref = None 

    opt = get_opt()
    test = False if opt.val else True
    single = True if opt.single else False

    gen_folder = opt.gen
    gt_folder = opt.gt

    if opt.noref or (opt.gen and not opt.gt):
        metrics_str = calc_metrics_without_gt(os.path.abspath(gen_folder))
        print_write(gen_folder, '', metrics_str)
    else:
        metrics_str = calc_metrics_with_gt(os.path.abspath(gen_folder), os.path.abspath(gt_folder))
        print_write(gen_folder, gt_folder, metrics_str)
