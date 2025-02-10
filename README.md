# A Content-Style Control Network with Style Contrastive Learning for Underwater Image Enhancement

This is an implement of the CSC-SCL, "[A  Content-Style Control Network with Style Contrastive Learning for Underwater Image  Enhancement](https://link.springer.com/article/10.1007/s00530-024-01642-z?utm_source=rct_congratemailt&utm_medium=email&utm_campaign=nonoa_20250109&utm_content=10.1007%2Fs00530-024-01642-z "link to the paper")", Zhenguang Wang, Huanjie Tao, Hui Zhou,  Yishi Deng, Ping Zhou.

## Overview



<img src=".\Overview.png" alt="Overview" style="zoom: 60%;" />

## Preparation

### Prerequisites

- Linux or macOS
- Python 3.9 to Python 3.11
- NVIDIA GPU + CUDA CuDNN
- MATLAB Runtime R2023b

### Installing Python Dependencies

Run the following commands in the project's root directory:

````bash
pip install -r requirements.txt
cd packages
pip install ./calc_uciqe
````

### MATLAB Runtime

MATLAB Runtime is required for calculating UCIQE. Please install and configure MATLAB Runtime R2023b by following the instructions at: https://ww2.mathworks.cn/help/compiler/install-the-matlab-runtime.html.

Note: Using MATLAB Runtime to calculate UCIQE requires Python version 3.9 to 3.11.

## Testing

1. Download the pretrained model from the following link: https://pan.baidu.com/s/1IfPyrHzexqvt-7-0PRyyJg?pwd=5r9x, Extract and place it in the `checkpoints` directory:
   
   ```bash
   unzip pretrained_models.zip -d ./checkpoints/pretrained_models
   ```
   
2. Prepare the dataset and place it in the `datasets` directory with the following structure:
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

   Alternatively, download our UIEB dataset split from the following link: https://pan.baidu.com/s/1ngu4L0dowEQkfidg5Pv4SA?pwd=5izk.

   ```bash
   unzip UIEB_dataset.zip -d ./datasets/UIEB_dataset
   ```

3. Run the testing script:
   ```bash
   ./scripts/inference_calc_metrics.sh pretrained_models UIEB_dataset
   ```

   Or run the scripts step by step for inference and metric calculation:
   
   ```bash
   # inference
   python test.py --dataroot ./datasets/UIEB_dataset/testA --name pretrained_models --model test --load_size 256 --preprocess resize --dataset_mode single --model_suffix _A --no_dropout --epoch 800 --results_dir ./results --netG 'resnet_9blocks_cc_up_sc' --gpu_ids 0
   # calculate metrics
   python calc_metrics.py --gen ./results/pretrained_models/test_800/images --gt ./datasets/UIEB_dataset/testB_gt --single
   ```

You can find the enhanced images in the `results` directory.

## Training

First, train a feature extractor:

```bash
python ./contrast/pretrain_extractor.py
```

Then, train the underwater image enhancement model and specify the feature extractor path.

```bash
python train.py --dataroot ./datasets/UIEB_HCLR --name uieb_perc1rn800_cont1e800_cc_bn_up_sc_aug_800 --model cycle_gan --netG resnet_9blocks_cc_up_sc --has_perc --has_cont --cont extractor --extr_path "./checkpoints/type_extractor/uieb_cc_sc_400/400_net_E.pth" --load_size 256 --preprocess augmentation --n_epochs 300 --n_epochs_decay 500 --save_epoch_freq 50 --batch_size 2 --gpu_ids 0
```

You can find the saved model parameters in the `checkpoints` directory.

## Acknowledgement

This repository contains modified codes from:

* [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
* [Image-quality-measure-method](https://github.com/Owen718/Image-quality-measure-method)
* [DAC](https://github.com/sangrokleeeeee/DAC)
* [TACL](https://github.com/Jzy2017/TACL)
