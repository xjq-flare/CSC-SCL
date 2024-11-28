"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""

import calc_uciqe
import calc_niqe

import warnings
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import os
import torch
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group

import numpy as np
import random
from data.verify_dataset import VerifyDataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import save_image
from utils.metrics import calc_all_metrics


warnings.filterwarnings("ignore")


class Trainer:
    def __init__(self, model, train_data, verify_data, opt, rank: int):
        self.model = model
        self.train_data = train_data
        self.verify_data = verify_data
        self.opt = opt
        self.rank = rank

    def train(self):
        if self.rank == 0:
            visualizer = Visualizer(self.opt)  # create a visualizer that display/save images and plots
            val_file_path = f"./results/{self.opt.name}"
            if not os.path.exists(val_file_path):
                os.makedirs(val_file_path)
            val_file = open(os.path.join(val_file_path, "val_log.txt"), 'w', encoding='utf-8')

        total_iters = 0  # the total number of training iterations
        dataset_size = len(self.train_data)
        for epoch in range(self.opt.epoch_count,
                           self.opt.n_epochs + self.opt.n_epochs_decay + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
            if self.rank == 0:
                print('epoch:', epoch, ' ', end='')
            self.train_data.dataloader.sampler.set_epoch(epoch)
            epoch_start_time = time.time()  # timer for entire epoch
            iter_data_time = time.time()  # timer for data loading per iteration
            epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
            if self.rank == 0:
                visualizer.reset()  # reset the visualizer: make sure it saves the results to HTML at least once every epoch
            self.model.update_learning_rate()  # update learning rates in the beginning of every epoch.
            # objs = utils.AverageMeter()
            for i, data in enumerate(self.train_data):  # inner loop within one epoch
                iter_start_time = time.time()  # timer for computation per iteration
                if total_iters % self.opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time

                total_iters += self.opt.batch_size
                epoch_iter += self.opt.batch_size
                self.model.set_input(data)  # unpack data from dataset and apply preprocessing
                self.model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

                # if self.rank == 0:
                #     objs.update(loss_G.data.item(), self.opt.batch_size)

                if self.rank == 0 and total_iters % self.opt.display_freq == 0:  # display images on visdom and save images to a HTML file
                    save_result = total_iters % self.opt.update_html_freq == 0
                    self.model.compute_visuals()
                    visualizer.display_current_results(self.model.get_current_visuals(), epoch, save_result)

                if self.rank == 0 and total_iters % self.opt.print_freq == 0:  # print training losses and save logging information to the disk
                    losses = self.model.get_current_losses()
                    t_comp = (time.time() - iter_start_time) / self.opt.batch_size
                    visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                    if self.opt.display_id > 0:
                        visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

                if self.rank == 0 and total_iters % self.opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
                    print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                    save_suffix = 'iter_%d' % total_iters if self.opt.save_by_iter else 'latest'
                    self.model.save_networks(save_suffix)

                iter_data_time = time.time()
            if epoch % self.opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
                print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                self.model.save_networks('latest')
                self.model.save_networks(epoch)

            # epoch training end
            if self.rank == 0:
                # print(f"average generator loss: {objs.avg}")
                print('End of epoch %d / %d \t Time Taken: %d sec' % (
                    epoch, self.opt.n_epochs + self.opt.n_epochs_decay, time.time() - epoch_start_time))

            if epoch % self.opt.verify_epoch_freq == 0:
                with torch.no_grad():
                    self.model.netG_A.eval()
                    for i, data in enumerate(self.verify_data):
                        gen_img = self.model.netG_A(data['A'].to(self.model.device))
                        save_dir = f"./results/{self.opt.name}/valB"
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)
                        for index in range(0, gen_img.shape[0]):
                            image_name = os.path.split(data['A_paths'][index])[-1]
                            save_image((gen_img[index].squeeze() + 1.0) * 0.5, os.path.join(save_dir, image_name))

                self.model.netG_A.train()
                if self.rank == 0:
                    psnr, ssim, uiqm, _, _, _, uciqe, niqe = calc_all_metrics(
                        os.path.abspath(f"./results/{self.opt.name}/valB"),
                        os.path.abspath(os.path.join(self.opt.dataroot, 'valB_gt')),
                        (256, 256))
                    print(f"validation PSNR: {psnr} | SSIM: {ssim} | UIQM: {uiqm} | UCIQE: {uciqe} | NIQE: {niqe}")
                    val_file.write(f"validation PSNR: {psnr} | SSIM: {ssim} | UIQM: {uiqm} | UCIQE: {uciqe} | NIQE: {niqe}\n")


def ddp_setup(rank, world_size, gpu_ids):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    # init_process_group(backend="gloo", rank=rank, world_size=world_size)
    torch.cuda.set_device(gpu_ids[rank])


def main(rank: int, world_size: int, opt):
    random.seed(1024)
    np.random.seed(1024)
    torch.manual_seed(1024)

    ddp_setup(rank, world_size, opt.gpu_ids)
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options

    batch_size = opt.batch_size // world_size
    verify_dataset = VerifyDataset(os.path.join(opt.dataroot, 'valA'))
    verify_dataloader = DataLoader(verify_dataset,
                                   batch_size=batch_size,
                                   num_workers=int(opt.num_threads),
                                   pin_memory=True,
                                   sampler=DistributedSampler(verify_dataset, shuffle=False))


    distributed_sampler = DistributedSampler(dataset.dataset,
                                             shuffle=not opt.serial_batches)

    dataset.dataloader = DataLoader(dataset.dataset,
                                    batch_size=batch_size,
                                    num_workers=int(opt.num_threads),
                                    pin_memory=True,
                                    sampler=distributed_sampler,
                                    drop_last=opt.drop_last)
    print(f"Rank: {torch.distributed.get_rank()} changing dataloader and sampler...")

    dataset_size = len(dataset)  # get the number of images in the dataset.
    print('Rank: %d The number of training images about %d on each rank, all = %d' %
          (rank, len(dataset.dataloader) * batch_size, dataset_size))
    model = create_model(opt, rank)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    trainer = Trainer(model, dataset, verify_dataloader, opt, rank)
    trainer.train()
    destroy_process_group()


if __name__ == '__main__':
    opt = TrainOptions().parse()  # get training options
    world_size = len(opt.gpu_ids)
    mp.spawn(main, args=(world_size, opt), nprocs=world_size)
