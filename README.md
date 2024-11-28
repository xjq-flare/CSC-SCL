安装matlab runtime

https://ww2.mathworks.cn/help/compiler/install-the-matlab-runtime.html

安装 calc_uciqe和calc_niqe

unzip calc_uciqe.zip

pip install calc_uciqe



# A Content-Style Control Network with Style Contrastive Learning for Underwater Image Enhancement

This is an implement of the CSC-SCL, "A  Content-Style Control Network with Style Contrastive Learning for Underwater Image  Enhancement", Zhenguang Wang, Huanjie Tao, Hui Zhou,  Yishi Deng, Ping Zhou.

## Overview



<img src=".\Overview.png" alt="Overview" style="zoom: 80%;" />

## Preparation

### Prerequisites

- Linux or macOS
- Python 3.9 到 Python 3.11
- NVIDIA GPU + CUDA CuDNN
- MATLAB Runtime R2023b

### 安装python依赖

在项目根目录执行以下命令：

````bash
pip install -r requirements.txt
cd packages
pip install ./calc_uciqe
````

### MATLAB Runtime

计算uciqe时需要使用MATLAB Runtime，请按照链接https://ww2.mathworks.cn/help/compiler/install-the-matlab-runtime.html安装并配置MATLAB Runtime R2023b。

## Testing

1. 下载预训练模型，链接为：https://pan.baidu.com/s/1IfPyrHzexqvt-7-0PRyyJg?pwd=5r9x
   解压并放到`checkpoints`目录下：

   ```bash
   unzip pretrained_models.zip -d ./checkpoints/pretrained_models
   ```

2. 准备数据集，并放到`datasets`目录下，目录结构如下：
   ```bash
   UIEB_HCLR/
   ├── testA
       ├── 0.png
       ├── 1.png
       └── 2.png
   ├── testB_gt
   ├── trainA
   ├── trainB
   ├── valA
   └── valB_gt
   ```

   或下载我们的UIEB数据集划分，链接为：https://pan.baidu.com/s/1ngu4L0dowEQkfidg5Pv4SA?pwd=5izk

   ```bash
   unzip UIEB_dataset.zip -d ./datasets/UIEB_dataset
   ```

3. 运行测试脚本：
   ```bash
   ./scripts/inference_calc_metrics.sh pretrained_models UIEB_dataset
   ```

   或依次运行脚本进行推理和计算评价指标：
   
   ```bash
   # inference
   python test.py --dataroot ./datasets/UIEB_dataset/testA --name pretrained_models --model test --load_size 256 --preprocess resize --dataset_mode single --model_suffix _A --no_dropout --epoch 800 --results_dir ./results --netG 'resnet_9blocks_cc_up_sc' --gpu_ids 0
   # calculate metrics
   python calc_metrics.py --gen ./results/pretrained_models/test_800/images --gt ./datasets/UIEB_dataset/testB_gt --single
   ```

## Training

首先，训练一个特征提取器。

```bash
python ./contrast/pretrain_extractor.py
```

然后训练水下图像增强模型。

```bash
python train.py --dataroot ./datasets/UIEB_HCLR --name uieb_perc1rn800_cont1e800_cc_bn_up_sc_aug_800 --model cycle_gan --netG resnet_9blocks_cc_up_sc --has_perc --has_cont --cont extractor --extr_path "./checkpoints/type_extractor/uieb_cc_sc_400/400_net_E.pth" --load_size 256 --preprocess augmentation --n_epochs 300 --n_epochs_decay 500 --save_epoch_freq 50 --batch_size 2 --gpu_ids 0
```

## Acknowledgement

This repository contains modified codes from:

* [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
* [Image-quality-measure-method](https://github.com/Owen718/Image-quality-measure-method)
* [DAC](https://github.com/sangrokleeeeee/DAC)
