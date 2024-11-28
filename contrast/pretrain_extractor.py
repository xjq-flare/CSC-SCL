import os
import time
from munch import Munch
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image, ImageFilter
from torchvision.models import vgg
import functools

import sys
sys.path.append('../')
from models.networks import DownSampleLayer, ResnetBlock, ResnetBlockSC, SCNorm


opt = Munch()

##### config

opt.dataroot = '../datasets/UIEB_dataset/'
opt.save_dir = '../checkpoints/type_extractor/uieb_cc_sc_400/'

opt.mosaic = False
opt.mosaic_size = (16, 16)
opt.blur = False
opt.use_vgg = False
opt.adv_norm = 'cc_sc'

opt.n_epochs = 100
opt.n_epochs_decay = 300
opt.lr = 0.0002
opt.batch_size = 32

opt.print_freq = 30
opt.save_freq = 50
opt.verify_freq = 10

opt.num_workers = 8
opt.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
opt.log_file_path = os.path.join(opt.save_dir, 'log.txt')

##### end config

if not os.path.exists(opt.save_dir):
    os.makedirs(opt.save_dir)

class GaussianBlur:
    def __init__(self, window_size):
        self.window_size = window_size

    def __call__(self, img):
        return img.filter(ImageFilter.GaussianBlur(self.window_size))

class Mosaic:
    def __init__(self, a=16, b=16):
        self.a = a
        self.b = b

    def __call__(self, input_tensor):
        c, H, W = input_tensor.size()

        m, n = H // self.a, W // self.b

        input_tensor = input_tensor.view(c, m, self.a, n, self.b)
        input_tensor = input_tensor.permute(1, 3, 0, 2, 4).contiguous()
        input_tensor = input_tensor.view(m * n, c, self.a, self.b)

        indices = torch.randperm(m * n)
        input_tensor = input_tensor[indices]

        input_tensor = input_tensor.view(m, n, c, self.a, self.b)
        input_tensor = input_tensor.permute(2, 0, 3, 1, 4).contiguous()
        input_tensor = input_tensor.view(c, H, W)

        return input_tensor

class Log:
    def __init__(self, log_file_path):
        self.log_file = open(log_file_path, 'a', encoding='utf-8')
        
        self.print(str(datetime.now()) + '\n')
        for key, value in opt.items():
            self.print(f"{key} = {value}")

    def print(self, message):
        print(message)
        self.log_file.write(message + '\n')

    def __del__(self):
        self.log_file.close()


class Vgg19(nn.Module):
    def __init__(self):
        super(Vgg19, self).__init__()
        self.vgg = vgg.vgg19(weights=vgg.VGG19_Weights.DEFAULT).features
        for param in self.vgg.parameters():
            param.requires_grad_(False)
        self.flatten = nn.Flatten(start_dim=1)
        self.head = nn.Sequential(
            nn.Linear(7*7*512, 1),
            nn.Sigmoid())
        
    def forward(self, img):
        out = self.vgg(img)
        out = self.flatten(out)
        return self.head(out)
        


class Extractor(nn.Module):
    def __init__(self, norm_layer=nn.InstanceNorm2d, use_dropout=False, ngf=64, padding_type='reflect', adv_norm='cc_sc'):
        super(Extractor, self).__init__()
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(3, ngf, kernel_size=7, padding=0, stride=2, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(inplace=True),
                 nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        mult = 1
        # add ResNet blocks
        if adv_norm == '':
            model += [
                ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                            use_bias=use_bias)]
        elif 'sc' in adv_norm and 'sc(add)' not in adv_norm:
            model += [
                ResnetBlockSC(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                use_bias=use_bias)]
        elif 'sc(add)' in adv_norm:
            model += [
                ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                            use_bias=use_bias), SCNorm(ngf * mult)]

        for _ in range(3):
            # add downsample layer
            if 'cc' in adv_norm:
                if 'cc_' in adv_norm or 'cc(bn)' in adv_norm:
                    model.append(DownSampleLayer(ngf * mult, ngf * mult * 2, norm_layer, use_bias, residual_norm='bn'))
                elif 'cc(in)' in adv_norm:
                    model.append(DownSampleLayer(ngf * mult, ngf * mult * 2, norm_layer, use_bias, residual_norm='in'))
            else:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                            norm_layer(ngf * mult * 2),
                            nn.ReLU(True)]
        
            mult *= 2

            # add ResNet blocks
            if adv_norm == '':
                model += [
                    ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                use_bias=use_bias)]
            elif 'sc' in adv_norm and 'sc(add)' not in adv_norm:
                model += [
                    ResnetBlockSC(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                use_bias=use_bias)]
            elif 'sc(add)' in adv_norm:
                model += [
                    ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                use_bias=use_bias), SCNorm(ngf * mult)]

        model.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.backbone = nn.Sequential(*model)
        self.flatten = nn.Flatten()
        self.head = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.backbone(input)
        output = self.flatten(output)
        return self.head(output)


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


class DataSet:
    IMG_EXTENSIONS = [
            '.jpg', '.JPG', '.jpeg', '.JPEG',
            '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
            '.tif', '.TIF', '.tiff', '.TIFF',
        ]

    def __init__(self, dir_A_list, dir_B_list, opt):
        self.dir_A_list = dir_A_list
        self.dir_B_list = dir_B_list
        self.opt = opt

        self.paths_A = []
        for dir_A in self.dir_A_list:
            self.paths_A += sorted(self.make_dataset(dir_A))
        self.paths_B = []
        for dir_B in self.dir_B_list:
            self.paths_B += sorted(self.make_dataset(dir_B))
        self.num_A = len(self.paths_A)
        self.paths = self.paths_A + self.paths_B
        if not opt.use_vgg:
            img_size = 256
            mean = (0.5, 0.5, 0.5)
            std = (0.5, 0.5, 0.5)
        else:
            img_size = 224
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
        transform_list = [
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]
        if opt.blur:
            transform_list.insert(2, GaussianBlur(img_size))
        if opt.mosaic:
            transform_list.append(Mosaic(*opt.mosaic_size))
        self.transform = transforms.Compose(transform_list)

    @classmethod
    def is_image_file(cls, filename):
        return any(filename.endswith(extension) for extension in cls.IMG_EXTENSIONS)

    def make_dataset(self, dir):
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if self.is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
        return images

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path).convert('RGB')
        if index < self.num_A:
            label = 0
        else:
            label = 1

        return {'img': self.transform(img), 'label': label, 'path': path}

    def __len__(self):
        return len(self.paths)
    

def lambda_rule(epoch):
    lamda = 1.0 - max(0, epoch - opt.n_epochs) / float(opt.n_epochs_decay + 1)
    return lamda


# start train

log = Log(opt.log_file_path)

if not opt.use_vgg:
    model = Extractor(adv_norm=opt.adv_norm)
    model.to(opt.device)
    init_weights(model)
else:
    model = Vgg19()
    model.to(opt.device)
    nn.init.normal_(model.head[0].weight.data, 0.0, 0.02)
    nn.init.constant_(model.head[0].bias.data, 0.0)

num_params = 0
for param in model.parameters():
    num_params += param.numel()
log.print('Total number of parameters : %.3f M' % (num_params / 1e6))


dataset = DataSet([os.path.join(opt.dataroot, 'trainA')], [os.path.join(opt.dataroot, 'trainB')], opt)
dataset_verify = DataSet([os.path.join(opt.dataroot, 'valA')], [os.path.join(opt.dataroot, 'valB_gt')], opt)

dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
dataloader_verify = DataLoader(dataset_verify, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

criterion = nn.BCELoss()

optimizer = optim.Adam(model.parameters(), lr=opt.lr)
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

for epoch in range(1, opt.n_epochs + opt.n_epochs_decay + 1):
    total_iters = 0
    model.train()
    epoch_start = time.time()
    iter_data_time = time.time()
    for total_iters, data in enumerate(dataloader, 1):
        iter_start_time = time.time()
        if total_iters % opt.print_freq == 0:
            t_data = iter_start_time - iter_data_time
        imgs, labels = data['img'].to(opt.device), data['label'].to(opt.device)
        outs = model(imgs)
        # loss = criterion(outs, one_hot(labels, num_classes=2).float())
        loss = criterion(outs.squeeze(), labels.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        t_comp = time.time() - iter_start_time
        if total_iters % opt.print_freq == 0:
            log.print(f"epoch: {epoch}, iters: {total_iters}, time: {t_comp:.6f}, data: {t_data:.6f}, loss: {loss.item():.6f}")
        iter_data_time = time.time()
    
    log.print(f"End of epoch {epoch} / {opt.n_epochs + opt.n_epochs_decay} Time taken: {time.time() - epoch_start:.6f} sec")
    old_lr = scheduler.get_last_lr()
    scheduler.step()
    log.print(f"learning rate: {old_lr[0]:.6f} -> {scheduler.get_last_lr()[0]:.6f}")
    if epoch % opt.save_freq == 0:
        print(f"saving the model at the end of epoch: {epoch}")
        if not os.path.exists(opt.save_dir):
            os.makedirs(opt.save_dir)
        torch.save(model.state_dict(), os.path.join(opt.save_dir, f"{epoch}_net_E.pth"))
    if epoch % opt.verify_freq == 0:
        correct_predictions = 0
        total_samples = 0
        model.eval()
        with torch.no_grad():
            for data in dataloader_verify:
                imgs, labels = data['img'].to(opt.device), data['label'].to(opt.device)
                outs = model(imgs)
                # _, predicted = torch.max(outs, dim=1)
                predicted = outs.ge(0.5).squeeze()
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
            accuracy = correct_predictions / total_samples
            log.print(f'Validation Accuracy: {accuracy * 100:.2f}%')
        model.train()
