# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

import torch.nn.functional as F
import numpy as np
from unet import UNet, UNet_v2
from torchvision import models


def size_calculator(input_size, padding=1, kernel=3, stride=1, layer=1):
    for i in range(layer):
        input_size = (input_size + (2*padding-kernel)) // stride + 1
    return input_size



class COTTON(nn.Module):
    def __init__(self, config):
        super(COTTON, self).__init__()
        self.d_sample = config['TRAINING_CONFIG']['D_SAMPLE']
        self.parsing_G = UNet_v2(n_channels=8, n_classes=15, bilinear=False)
        self.tryon_G = FashionOn_MultiG(ch_in=24, z_num=64, repeat_num=6,hidden_num=128, shape=config['TRAINING_CONFIG']['RESOLUTION'])

        
    def forward(self, c_img, parsing_masked, human_masked, human_pose):
        # Generate parsing prediction
        parsing_input = torch.cat([c_img, parsing_masked, human_pose], dim=1)
        if self.d_sample != 1:
            parsing_input = F.interpolate(parsing_input, scale_factor=1/self.d_sample)

        parsing_pred = self.parsing_G(parsing_input)
        if self.d_sample != 1:
            parsing_pred = F.interpolate(parsing_pred, scale_factor=self.d_sample)

        parsing_pred_hard = F.gumbel_softmax(parsing_pred, hard=True, dim=1)
        tryon_input = torch.cat([human_masked, parsing_pred_hard, c_img], dim=1)
        tryon_img_fake = self.tryon_G(tryon_input)

        return parsing_pred, parsing_pred_hard, tryon_img_fake


class Discriminator(nn.Module):
    def __init__(self, input_nc=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(input_nc*2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)
        
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return x.view(x.shape[0], -1)

class FashionOn_MultiG(nn.Module):
    # extend to MultiG based on GeneratorCNN_Pose_UAEAfterResidual_256
    def block(self, ch_in, ch_out, kernel, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel, stride, padding),
            nn.ReLU(True),
            nn.Conv2d(ch_out, ch_out, kernel, stride, padding),
            nn.ReLU(True)
        )
    
    def block_one(self, ch_in, ch_out, kernel, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel, stride, padding),
            nn.ReLU(True)
        )
    
    def conv(self, ch_in, ch_out, kernel, stride=1, padding=1):
        return nn.Conv2d(ch_in, ch_out, kernel, stride, padding)
        
    def fc(self, ch_in, ch_out):
        return nn.Linear(ch_in, ch_out)
    
    def upsample(self, ch_in, kernel_size, stride=2, padding=1, output_padding=1):
        ch_out = ch_in
        norm_layer = torch.nn.GroupNorm
        group_num = 32
        return nn.Sequential(
            nn.ConvTranspose2d(ch_in, ch_out, kernel_size, stride, padding, output_padding),
            norm_layer(group_num, ch_out),
            nn.ReLU(True)
        )

    #(ch_in=9, z_num=64, repeat_num=6,hidden_num=128)
    def __init__(self, ch_in, z_num, repeat_num, hidden_num=128, shape=(1024, 768)):
        super(FashionOn_MultiG, self).__init__()
        self.min_fea_map_H = shape[0] // 2**repeat_num
        self.min_fea_map_W = shape[1] // 2**repeat_num
        self.z_num = z_num 
        self.hidden_num = hidden_num 
        self.repeat_num = repeat_num
        
        # ===================== #
        #   global Downsample   #
        # ===================== #

        self.G_block_1 = self.block_one(ch_in, self.hidden_num, 3, 1)

        self.G_block1 = self.block(self.hidden_num, 128, 3, 1)
        self.G_block2 = self.block(256, 256, 3, 1)
        self.G_block_one1 = self.block_one(128, 256, 3, 2)

        # ========== #
        #    local   #
        # ========== #

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

        self.block_1 = self.block_one(ch_in, self.hidden_num, 3, 1)

        self.block1 = self.block(self.hidden_num, 128, 3, 1)
        self.block2 = self.block(256, 256, 3, 1)
        self.block3 = self.block(384, 384, 3, 1)
        self.block4 = self.block(512, 512, 3, 1)
        self.block5 = self.block(640, 640, 3, 1)
        self.block6 = self.block(768, 768, 3, 1)
            
        self.block_one1 = self.block_one(128, 256, 3, 2)
        self.block_one2 = self.block_one(256, 384, 3, 2)
        self.block_one3 = self.block_one(384, 512, 3, 2)
        self.block_one4 = self.block_one(512, 640, 3, 2)    # 20210504 modify to make output be [bz, 640, 20, "16"]
        self.block_one5 = self.block_one(640, 768, 3, 2)
        
        self.fc1 = self.fc(self.min_fea_map_H * self.min_fea_map_W * 768, self.z_num)
        self.fc2 = self.fc(self.z_num, self.min_fea_map_H * self.min_fea_map_W * self.hidden_num)
        
        self.block7 = self.block(896, 896, 3, 1)
        self.block8 = self.block(1280, 1280, 3, 1)
        self.block9 = self.block(1024, 1024, 3, 1)
        self.block10 = self.block(768, 768, 3, 1)
        self.block11 = self.block(512, 512, 3, 1)
        self.block12 = self.block(256, 256, 3, 1)
        
        self.block_one6 = self.block_one(896, 640, 1, 1, padding=0)
        self.block_one7 = self.block_one(1280, 512, 1, 1, padding=0)
        self.block_one8 = self.block_one(1024, 384, 1, 1, padding=0)
        self.block_one9 = self.block_one(768, 256, 1, 1, padding=0)
        self.block_one10 = self.block_one(512, 128, 1, 1, padding=0)
        
        
        
        # self.upscale = nn.Upsample(scale_factor=2)

        self.upsample7 = self.upsample(ch_in=896, kernel_size=3)
        self.upsample8 = self.upsample(ch_in=1280, kernel_size=3)   # set kernel size to be 2 for matching the concatenated feature size downsample
        self.upsample9 = self.upsample(ch_in=1024, kernel_size=3)
        self.upsample10 = self.upsample(ch_in=768, kernel_size=3)
        self.upsample11 = self.upsample(ch_in=512, kernel_size=3)

        # ================== #
        #   global Upsample  #
        # ================== #

        # self.Resnet_block = nn.Sequential(
        #                     ResnetBlock(ch_in, padding_type=padding_type, norm_layer=norm_layer, use_dropout=False, use_bias=use_bias),
        #                     ResnetBlock(ch_in, padding_type=padding_type, norm_layer=norm_layer, use_dropout=False, use_bias=use_bias),
        #                     ResnetBlock(ch_in, padding_type=padding_type, norm_layer=norm_layer, use_dropout=False, use_bias=use_bias)
        #                     )
        self.G_upsample12 = self.upsample(ch_in=256, kernel_size=3)
        self.G_block_one11 = self.block_one(256, 128, 1, 1, padding=0)
        self.G_block13 = self.block(128, 128, 3, 1)
        self.conv_last = self.conv(128, 3, 3, 1) 

    def forward(self, x):
        # ===================== #
        #   global Downsample   #
        # ===================== #
        y = self.G_block_1(x)
        # x: torch.Size([1, 128, 640, 480])

        # 1st encoding layer
        res = y
        y = self.G_block1(y)
        # x: torch.Size([1, 128, 640, 480])
        y = y + res
        
        # encoder_layer_list.append(y)
        y = self.G_block_one1(y)
        # x: torch.Size([1, 256, 320, 240])
        
        # 2nd encoding layer
        res = y
        y = self.G_block2(y)
        # x: torch.Size([1, 256, 320, 240])
        y = y + res

        # ========== #
        #    local   #
        # ========== #

        x = self.downsample(x)
        # x: torch.Size([1, 10, 320, 240])
        encoder_layer_list = []
        x = self.block_1(x)
        # x: torch.Size([1, 128, 320, 240])

        # 1st encoding layer
        res = x
        x = self.block1(x)
        # x: torch.Size([1, 128, 320, 240])
        x = x + res
        
        encoder_layer_list.append(x)
        x = self.block_one1(x)
        # x: torch.Size([1, 256, 160, 120])
        
        # 2nd encoding layer
        res = x
        x = self.block2(x)
        # x: torch.Size([1, 256, 160, 120])
        x = x + res
        
        encoder_layer_list.append(x)
        x = self.block_one2(x)
        # x: torch.Size([1, 384, 80, 60])
        
        # 3rd encoding layer
        res = x
        x = self.block3(x)
        # x: torch.Size([1, 384, 80, 60])
        x = x + res
        
        encoder_layer_list.append(x)
        x = self.block_one3(x)
        # x: torch.Size([1, 512, 40, 30])
        
        # 4th encoding layer
        res = x
        x = self.block4(x)
        # x: torch.Size([1, 512, 40, 30])
        x = x + res
        
        encoder_layer_list.append(x)
        x = self.block_one4(x)
        # x: torch.Size([1, 640, 20, 16])
        
        # 5th encoding layer
        res = x
        x = self.block5(x)
        # x: torch.Size([1, 640, 20, 16])
        x = x + res
        
        encoder_layer_list.append(x)
        x = self.block_one5(x)
        # x: torch.Size([1, 768, 10, 8])
        
        # 6th encoding layer
        res = x
        x = self.block6(x)
        # x: torch.Size([1, 768, 10, 8])
        x = x + res

        encoder_layer_list.append(x)
        x = x.view(-1, self.min_fea_map_H * self.min_fea_map_W * 768)
        x = self.fc1(x)
        # x: torch.Size([1, 64])
        z = x
        
        x = self.fc2(z)
        # x: torch.Size([1, 10240])
        x = x.view(-1, self.hidden_num, self.min_fea_map_H, self.min_fea_map_W)
        # x: torch.Size([1, 128, 10, 8])

        # 1st decoding layer
        x = torch.cat([x, encoder_layer_list[5]], dim=1)
        # x: torch.Size([1, 896, 10, 8])

        res = x
        x = self.block7(x)
        # x: torch.Size([1, 896, 10, 8])
        x = x + res
        x = self.upsample7(x)
        # x: torch.Size([1, 896, 20, 16])
        x = self.block_one6(x)
        # x: torch.Size([1, 640, 20, 16])
        
        # 2nd decoding layer
        x = torch.cat([x, encoder_layer_list[4]], dim=1)
        # x: torch.Size([1, 1280, 20, 16])
        res = x
        x = self.block8(x)
        # x: torch.Size([1, 1280, 20, 16])
        x = x + res
        x = self.upsample8(x)
        # x: torch.Size([1, 1280, 40, 30])
        x = self.block_one7(x)
        # x: torch.Size([1, 512, 40, 30])

        # 3rd decoding layer
        x = torch.cat([x, encoder_layer_list[3]], dim=1)
        # x: torch.Size([1, 1024, 40, 30])
        res = x
        x = self.block9(x)
        # x: torch.Size([1, 1024, 40, 30])
        x = x + res
        x = self.upsample9(x)
        # x: torch.Size([1, 1024, 80, 60])
        x = self.block_one8(x)
        # x: torch.Size([1, 384, 80, 60])

        # 4th decoding layer
        x = torch.cat([x, encoder_layer_list[2]], dim=1)
        # x: torch.Size([1, 768, 80, 60])
        res = x
        x = self.block10(x)
        # x: torch.Size([1, 768, 80, 60])
        x = x + res
        x = self.upsample10(x)
        # x: torch.Size([1, 768, 160, 120])
        x = self.block_one9(x)
        # x: torch.Size([1, 256, 160, 120])

        # 5th decoding layer
        x = torch.cat([x, encoder_layer_list[1]], dim=1)
        # x: torch.Size([1, 512, 160, 120])
        res = x
        x = self.block11(x)
        # x: torch.Size([1, 512, 160, 120])
        x = x + res
        x = self.upsample11(x)
        # x: torch.Size([1, 512, 320, 240])
        x = self.block_one10(x)
        # x: torch.Size([1, 128, 320, 240])

        # 6th decoding layer
        x = torch.cat([x, encoder_layer_list[0]], dim=1)
        # x: torch.Size([1, 256, 320, 240])
        res = x
        x = self.block12(x)
        # x: torch.Size([1, 256, 320, 240])
        x = x + res

        # ===================== #
        #    global Upsample    #
        # ===================== #
        x = x + y
        # x = y
        x = self.G_upsample12(x)
        # x: torch.Size([1, 256, 640, 480])
        x = self.G_block_one11(x)
        # x: torch.Size([1, 128, 640, 480])

        # 7th decoding layer
        res = x
        x = self.G_block13(x)
        # x: torch.Size([1, 128, 640, 480])
        x = x + res

        output = self.conv_last(x)
        # output: torch.Size([1, 3, 640, 480])
        output = nn.Tanh()(output)
        
        return output

class FashionOn_MultiD(nn.Module):
    # extend to MultiD based on DCGANDiscriminator_256
    def uniform(self, stdev, size):
        return np.random.uniform(
            low=-stdev * np.sqrt(3),
            high=stdev * np.sqrt(3),
            size=size
        ).astype('float32')
    
    def LeakyReLU(self, x, alpha=0.2):
        return torch.max(alpha*x, x)

    def conv2d(self, x, input_dim, filter_size, output_dim, gain=1, stride=1, padding=2):
        filter_values = self.uniform(
                self._weights_stdev,
                (output_dim, input_dim, filter_size, filter_size)
            )
        filter_values *= gain
        filters = torch.from_numpy(filter_values)
        biases = torch.from_numpy(np.zeros(output_dim, dtype='float32'))
        if self.use_gpu:
            filters = filters.cuda()
            biases = biases.cuda()
        result = nn.functional.conv2d(x, filters, biases, stride, padding)
        return result
        
    def LayerNorm(self, ch):
        norm_layer = torch.nn.GroupNorm
        group_num = 32
        return norm_layer(group_num, ch)
        # return nn.BatchNorm2d(ch)
        
    def __init__(self, num_D=3, bn=True, input_dim=3, dim=64, _weights_stdev=0.02, use_gpu=True, shape=(1024,768)):
        super(FashionOn_MultiD, self).__init__()
        self.num_D = num_D
        self.bn = bn
        self.input_dim = input_dim
        self.dim = dim
        self._weights_stdev = _weights_stdev
        self.use_gpu = use_gpu

        self.bn1 = self.LayerNorm(2*self.dim)
        self.bn2 = self.LayerNorm(4*self.dim)
        self.bn3 = self.LayerNorm(8*self.dim)
        self.bn4 = self.LayerNorm(8*self.dim)
        
        setattr(self, 'f_size0', size_calculator(shape[0], stride=2, layer=5) * size_calculator(shape[1], stride=2, layer=5))
        setattr(self, 'f_size1', size_calculator(shape[0], stride=2, layer=6) * size_calculator(shape[1], stride=2, layer=6))
        setattr(self, 'f_size2', size_calculator(shape[0], stride=2, layer=7) * size_calculator(shape[1], stride=2, layer=7))

        setattr(self, 'fc0', nn.Linear(8*self.dim*self.f_size0, 1)) 
        setattr(self, 'fc1', nn.Linear(8*self.dim*self.f_size1, 1))  
        setattr(self, 'fc2', nn.Linear(8*self.dim*self.f_size2, 1))  
        # self.fc1 = nn.Linear(8*6*8*self.dim, 1)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, x):
        output_list = []
        for i in range(self.num_D):
            output = x
            output = self.conv2d(output, self.input_dim, 5, self.dim, stride=2)
            output = self.LeakyReLU(output)
            output = self.conv2d(output, self.dim, 5, 2*self.dim, stride=2)
            
            if self.bn:
                output = self.bn1(output)

            output = self.LeakyReLU(output)
            output = self.conv2d(output, 2*self.dim, 5, 4*self.dim, stride=2)
            
            if self.bn:
                output = self.bn2(output)
            
            output = self.LeakyReLU(output)
            output = self.conv2d(output, 4*self.dim, 5, 8*self.dim, stride=2)
            
            if self.bn:
                output = self.bn3(output)
            
            output = self.LeakyReLU(output)
            output = self.conv2d(output, 8*self.dim, 5, 8*self.dim, stride=2)
            
            if self.bn:
                output = self.bn4(output)
            
            output = self.LeakyReLU(output) # i = 0 : [1, 512, 20, 15] ; i = 1 : [1, 512, 10, 8] ; i = 2 : [1, 512, 5, 4]
            f_size = getattr(self, 'f_size'+str(i))
            output = output.view(-1, 8*self.dim*f_size)
            fc = getattr(self, 'fc'+str(i))
            output = fc(output)
            output_list.append(output)

            x = self.downsample(x)

        return output_list

class FashionOn_VGGLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(FashionOn_VGGLoss, self).__init__()
        self.add_module('vgg', FashionOn_VGG19())
        self.vgg.cuda()
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)
        return G
        
    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        content_loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])

        # Compute loss
        style_loss = 0.0
        style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu3_4']), self.compute_gram(y_vgg['relu3_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu4_4']), self.compute_gram(y_vgg['relu4_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))


        return content_loss, style_loss

class FashionOn_VGG19(torch.nn.Module):
    def __init__(self):
        super(FashionOn_VGG19, self).__init__()
        features = models.vgg19(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(32, 34):
            self.relu5_3.add_module(str(x), features[x])

        for x in range(34, 36):
            self.relu5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
            'relu5_4': relu5_4,
        }
        return out


if __name__=="__main__":
    stage1 = ResnetGenerator(input_nc=3, output_nc=3, n_blocks=9)
    stage1.cuda()
    
    d = Discriminator(input_nc=20)
    d.cuda()
    