# -*- coding: utf-8 -*-
"""
@author: AIMMLab, National Chiao Tung University
"""
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import os
from PIL import Image

import numpy as np
import argparse
import imageio
import cv2

label_colours_merge = [(0,0,0)
        # 0=Background
        ,(255,0,0),(0,0,255),(85,51,0),(255,85,0),(0,119,221)
        # 1=Hat,  2=Hair,    3=Glove, 4=Sunglasses, 5=UpperClothes
        ,(0,0,85),(0,85,85),(51,170,221),(0,255,255),(85,255,170)
        # 6=Dress, 7=Coat, 8=Socks, 9=Pants, 10=Jumpsuits/Neck
        ,(170,255,85),(255,255,0),(255,170,0),(85,85,0),(0,255,255)
        # 11=Scarf, 12=Skirt, 13=Face, 14=LeftArm, 15=RightArm
        ,(85,255,170),(170,255,85),(255,255,0),(255,170,0)]
        # 16=LeftLeg, 17=RightLeg, 18=LeftShoe, 19=RightShoe

label_colours = [(0,0,0)
            # 0=Background
            ,(128,0,0),(255,0,0),(0,85,0),(170,0,51),(255,85,0)
            # 1=Hat,  2=Hair,    3=Glove, 4=Sunglasses, 5=UpperClothes
            ,(0,0,85),(0,119,221),(85,85,0),(0,85,85),(85,51,0)
            # 6=Dress, 7=Coat, 8=Socks, 9=Pants, 10=Jumpsuits/Neck
            ,(52,86,128),(0,128,0),(0,0,255),(51,170,221),(0,255,255)
            # 11=Scarf, 12=Skirt, 13=Face, 14=LeftArm, 15=RightArm
            ,(85,255,170),(170,255,85),(255,255,0),(255,170,0)]
            # 16=LeftLeg, 17=RightLeg, 18=LeftShoe, 19=RightShoe

label_colours_HC = [(0,0,0),(51,170,221),(255,85,0)]
                    # 0=Background, 1=Arm, 2=UpperClothes

label_colours_HCN = [(0,0,0),(51,170,221),(255,85,0),(85,51,0)]
                    # 0=Background, 1=Arm, 2=UpperClothes, 3=Neck

def save_checkpoint(model, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    # torch.save(model.module.cpu().state_dict(), save_path)  # train in multi GPUs and save only single GPU
    torch.save(model.cpu().state_dict(), save_path)  # train in x GPUs and save in x GPUs
    model.cuda()

def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print("parameters not found")
        return
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    print("load parameters success")
    model.cuda()
                
def imshow(img):
    '''
        input:  image (channels, h, w) or (h, w) in pytorch Tensor [-1,1]
    '''
    img = img.detach().cpu()/2+0.5 # unnormalize pytorch tensor
    npimg = img.numpy()
    if len(npimg.shape) == 4: #batch_size & RGB
        for i in range(npimg.shape[0]):
            plt.imshow(np.transpose(npimg[i], (1, 2, 0)))
            plt.show()
    elif len(npimg.shape) == 3: #RGB
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
    elif len(npimg.shape) == 2: #L
        plt.imshow(npimg)
        plt.show()

def save_image(image_numpy, image_path):
    if image_numpy.shape[2] == 1:
        image_numpy = image_numpy.reshape(image_numpy.shape[0], image_numpy.shape[1])
    imageio.imwrite(image_path, image_numpy)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def imsave(img, img_name):
    '''
        Used to save parsing image, 
            but need to be "visualize_parse" before this function
        input:  image (batch_size, channels, h, w) in pytorch Tensor [0,255]
    '''
    img = torchvision.utils.make_grid(img.detach().cpu())
    img = torchvision.transforms.ToPILImage()(img)
    #print("img.shape= ",img.shape)
    #img = img [0,:,:]
    #print("img.shape= ",img.shape)
    #img = Image.fromarray(img,'L')
    img.save(img_name)
    
def decode_labels(mask, labels, num_classes=20):
    """Decode batch of segmentation masks.
    
    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
    
    Returns:
      A batch with num_images RGB images of the same size as the input. 
    """
    outputs = np.array(labels)[mask].astype(np.uint8)
    outputs = np.transpose(outputs, (0,3,1,2))
    outputs = torch.from_numpy(outputs)
    return outputs

def visualize_parse(parse, use_argmax=False):
    '''
        input:  parsing mask (batch_size, channel, h, w) in pytorch Tensor [-1,1]
        output: parsing image (batch_size, h, w) in pytorch Tensor [0,255] with mode 'RGB'
    '''
    if use_argmax:
        parse = torch.argmax(parse.cpu(), dim=1)
    else:
        parse = parse.cpu().squeeze(1)
    parse_img = decode_labels(parse, label_colours_merge)

    return parse_img

def save_gray_parse(parse, img_name):
    '''
        input:  parsing mask (batch_size, channel, h, w) in pytorch Tensor [-1,1]
        output: parsing image (batch_size, h, w) in pytorch Tensor [0,19] with mode 'L'
    '''
    parse = torch.argmax(parse.cpu(), dim=1)
    #print("parse.shape= ", parse.shape)
    parse_img = parse.numpy()
    #print("parse_img.shape= ",parse_img.shape)
    img = parse_img[0,:,:]
    #print("img.shape= ",img.shape)
    #print("type(img)= ", type(img))
    #print("img.dtype= ", img.dtype)
    #print("value of img= ", img[192][50:80])
    img = Image.fromarray(img.astype('uint8'),'L')
    img.save(img_name)

def remap(tensor, condition=1):
    if condition == 1:
        return tensor / 2 + 0.5
    elif condition == 255:
        return ((tensor / 2 + 0.5) * 255).type(torch.ByteTensor)

# convert a tensor into a numpy array
def tensor2im(image_tensor, bytes=255.0, imtype=np.uint8):
    if image_tensor.dim() == 3:
        image_numpy = image_tensor.cpu().float().numpy()
    else:
        image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * bytes

    return image_numpy.astype(imtype)

def parse2im(image_tensor, bytes=255.0, imtype=np.uint8):
    if image_tensor.dim() == 3:
        image_numpy = image_tensor.cpu().float().numpy()
    else:
        image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = np.transpose(image_numpy, (1, 2, 0))

    return image_numpy.astype(imtype)

#def parallel_result_train(inC, inH, Hmask, Cmask, Hgt, Cgt, img_name):
def imsave_set(inH, inC, Hparsing, Cmask, Hpose, Hparsing_gen, img_name):
    inC = torchvision.utils.make_grid(inC.detach().cpu())
    inC = torchvision.transforms.ToPILImage()(inC)
    inH = torchvision.utils.make_grid(inH.detach().cpu())
    inH = torchvision.transforms.ToPILImage()(inH)
    Hparsing = torchvision.utils.make_grid(Hparsing.detach().cpu())
    Hparsing = torchvision.transforms.ToPILImage()(Hparsing)
    Cmask = torchvision.utils.make_grid(Cmask.detach().cpu())
    Cmask = torchvision.transforms.ToPILImage()(Cmask)
    Hpose = torchvision.utils.make_grid(Hpose.detach().cpu())
    Hpose = torchvision.transforms.ToPILImage()(Hpose)
    Hparsing_gen = torchvision.utils.make_grid(Hparsing_gen.detach().cpu())
    Hparsing_gen = torchvision.transforms.ToPILImage()(Hparsing_gen)

    new_im = Image.new('RGB', (inC.size[0]*6,inC.size[1]))
    new_im.paste(inH, (inC.size[0]*0,0))
    new_im.paste(inC, (inC.size[0]*1,0))
    new_im.paste(Hparsing, (inC.size[0]*2,0))
    new_im.paste(Cmask, (inC.size[0]*3,0))
    new_im.paste(Hpose, (inC.size[0]*4,0))
    new_im.paste(Hparsing_gen, (inC.size[0]*5,0))

    new_im.save(img_name)

def imsave_setALL(inH, inC, Hparsing, Cmask, Hpose, Hparsing_gen, result, img_name):
    inC = torchvision.utils.make_grid(inC.detach().cpu())
    inC = torchvision.transforms.ToPILImage()(inC)
    inH = torchvision.utils.make_grid(inH.detach().cpu())
    inH = torchvision.transforms.ToPILImage()(inH)
    Hparsing = torchvision.utils.make_grid(Hparsing.detach().cpu())
    Hparsing = torchvision.transforms.ToPILImage()(Hparsing)
    Cmask = torchvision.utils.make_grid(Cmask.detach().cpu())
    Cmask = torchvision.transforms.ToPILImage()(Cmask)
    Hpose = torchvision.utils.make_grid(Hpose.detach().cpu())
    Hpose = torchvision.transforms.ToPILImage()(Hpose)
    Hparsing_gen = torchvision.utils.make_grid(Hparsing_gen.detach().cpu())
    Hparsing_gen = torchvision.transforms.ToPILImage()(Hparsing_gen)
    result = torchvision.utils.make_grid(result.detach().cpu())
    result = torchvision.transforms.ToPILImage()(result)

    new_im = Image.new('RGB', (inC.size[0]*7,inC.size[1]))
    new_im.paste(inH, (inC.size[0]*0,0))
    new_im.paste(inC, (inC.size[0]*1,0))
    new_im.paste(Hparsing, (inC.size[0]*2,0))
    new_im.paste(Cmask, (inC.size[0]*3,0))
    new_im.paste(Hpose, (inC.size[0]*4,0))
    new_im.paste(Hparsing_gen, (inC.size[0]*5,0))
    new_im.paste(result, (inC.size[0]*6,0))

    new_im.save(img_name)

def imsave_Sample(inH, inParsing, HparsingMasked, Cmask, Hpose, Hparsing_gen, inHCparsing, humanMasked, inC, result, img_name):
    inC = torchvision.utils.make_grid(inC.detach().cpu())
    inC = torchvision.transforms.ToPILImage()(inC)
    inParsing = torchvision.utils.make_grid(inParsing.detach().cpu())
    inParsing = torchvision.transforms.ToPILImage()(inParsing)
    inH = torchvision.utils.make_grid(inH.detach().cpu())
    inH = torchvision.transforms.ToPILImage()(inH)
    HparsingMasked = torchvision.utils.make_grid(HparsingMasked.detach().cpu())
    HparsingMasked = torchvision.transforms.ToPILImage()(HparsingMasked)
    Cmask = torchvision.utils.make_grid(Cmask.detach().cpu())
    Cmask = torchvision.transforms.ToPILImage()(Cmask)
    Hpose = torchvision.utils.make_grid(Hpose.detach().cpu())
    Hpose = torchvision.transforms.ToPILImage()(Hpose)
    Hparsing_gen = torchvision.utils.make_grid(Hparsing_gen.detach().cpu())
    Hparsing_gen = torchvision.transforms.ToPILImage()(Hparsing_gen)
    inHCparsing = torchvision.utils.make_grid(inHCparsing.detach().cpu())
    inHCparsing = torchvision.transforms.ToPILImage()(inHCparsing)
    humanMasked = torchvision.utils.make_grid(humanMasked.detach().cpu())
    humanMasked = torchvision.transforms.ToPILImage()(humanMasked)
    result = torchvision.utils.make_grid(result.detach().cpu())
    result = torchvision.transforms.ToPILImage()(result)

    new_im = Image.new('RGB', (inC.size[0]*6,inC.size[1]*2))
    new_im.paste(inH, (inC.size[0]*0,0))
    new_im.paste(inParsing, (inC.size[0]*1,0))
    new_im.paste(HparsingMasked, (inC.size[0]*2,0))
    new_im.paste(Cmask, (inC.size[0]*3,0))
    new_im.paste(Hpose, (inC.size[0]*4,0))
    new_im.paste(Hparsing_gen, (inC.size[0]*5,0))

    new_im.paste(inHCparsing, (inC.size[0]*2,inC.size[1]))
    new_im.paste(humanMasked, (inC.size[0]*3,inC.size[1]))
    new_im.paste(inC, (inC.size[0]*4,inC.size[1]))
    new_im.paste(result, (inC.size[0]*5,inC.size[1]))

    new_im.save(img_name)


def imsave_2stages(human, clothes, human_parse, skeleton_i, cm_align, human_parse_masked, output1,\
                                    c_align, human_masked_wo_BG, output, img_name):
    
    human = torchvision.utils.make_grid(human.detach().cpu())
    human = torchvision.transforms.ToPILImage()(human)
    clothes = torchvision.utils.make_grid(clothes.detach().cpu())
    clothes = torchvision.transforms.ToPILImage()(clothes)
    human_parse = torchvision.utils.make_grid(human_parse.detach().cpu())
    human_parse = torchvision.transforms.ToPILImage()(human_parse)
    skeleton_i = torchvision.utils.make_grid(skeleton_i.detach().cpu())
    skeleton_i = torchvision.transforms.ToPILImage()(skeleton_i)
    cm_align = torchvision.utils.make_grid(cm_align.detach().cpu())
    cm_align = torchvision.transforms.ToPILImage()(cm_align)
    human_parse_masked = torchvision.utils.make_grid(human_parse_masked.detach().cpu())
    human_parse_masked = torchvision.transforms.ToPILImage()(human_parse_masked)
    output1 = torchvision.utils.make_grid(output1.detach().cpu())
    output1 = torchvision.transforms.ToPILImage()(output1)

    c_align = torchvision.utils.make_grid(c_align.detach().cpu())
    c_align = torchvision.transforms.ToPILImage()(c_align)
    human_masked_wo_BG = torchvision.utils.make_grid(human_masked_wo_BG.detach().cpu())
    human_masked_wo_BG = torchvision.transforms.ToPILImage()(human_masked_wo_BG)
    output = torchvision.utils.make_grid(output.detach().cpu())
    output = torchvision.transforms.ToPILImage()(output)

    new_im = Image.new('RGB', (human.size[0]*6,human.size[1]*3))
    # stage 1
    new_im.paste(human_parse_masked, (human.size[0]*2, 0))
    new_im.paste(skeleton_i, (human.size[0]*3, 0))
    new_im.paste(cm_align, (human.size[0]*4, 0))
    new_im.paste(output1, (human.size[0]*5, 0))
    # stage 2
    new_im.paste(human_masked_wo_BG, (human.size[0]*2, human.size[1]*1))
    new_im.paste(output1, (human.size[0]*3, human.size[1]*1))
    new_im.paste(c_align, (human.size[0]*4, human.size[1]*1))
    new_im.paste(output, (human.size[0]*5, human.size[1]*1))
    # lower most IO
    new_im.paste(human, (human.size[0]*0, human.size[1]*2))
    new_im.paste(human_parse, (human.size[0]*1, human.size[1]*2))
    new_im.paste(clothes, (human.size[0]*2, human.size[1]*2))
    new_im.paste(output1, (human.size[0]*3, human.size[1]*2))
    new_im.paste(c_align, (human.size[0]*4, human.size[1]*2))
    new_im.paste(output, (human.size[0]*5, human.size[1]*2))

    new_im.save(img_name)


def imsave_3stages(human, clothes, skeleton_i, cm_align, human_parse_masked, output1, parsing_c, c_align,\
                    warped_cloth, human_masked_w_BG, warped_cloth_for_stage2, output, img_name):
    
    human = torchvision.utils.make_grid(human.detach().cpu())
    human = torchvision.transforms.ToPILImage()(human)
    clothes = torchvision.utils.make_grid(clothes.detach().cpu())
    clothes = torchvision.transforms.ToPILImage()(clothes)
    skeleton_i = torchvision.utils.make_grid(skeleton_i.detach().cpu())
    skeleton_i = torchvision.transforms.ToPILImage()(skeleton_i)
    cm_align = torchvision.utils.make_grid(cm_align.detach().cpu())
    cm_align = torchvision.transforms.ToPILImage()(cm_align)
    human_parse_masked = torchvision.utils.make_grid(human_parse_masked.detach().cpu())
    human_parse_masked = torchvision.transforms.ToPILImage()(human_parse_masked)
    output1 = torchvision.utils.make_grid(output1.detach().cpu())
    output1 = torchvision.transforms.ToPILImage()(output1)

    parsing_c = torchvision.utils.make_grid(parsing_c.detach().cpu())
    parsing_c = torchvision.transforms.ToPILImage()(parsing_c)
    c_align = torchvision.utils.make_grid(c_align.detach().cpu())
    c_align = torchvision.transforms.ToPILImage()(c_align)
    warped_cloth = torchvision.utils.make_grid(warped_cloth.detach().cpu())
    warped_cloth = torchvision.transforms.ToPILImage()(warped_cloth)
    human_masked_w_BG = torchvision.utils.make_grid(human_masked_w_BG.detach().cpu())
    human_masked_w_BG = torchvision.transforms.ToPILImage()(human_masked_w_BG)
    warped_cloth_for_stage2 = torchvision.utils.make_grid(warped_cloth_for_stage2.detach().cpu())
    warped_cloth_for_stage2 = torchvision.transforms.ToPILImage()(warped_cloth_for_stage2)
    output = torchvision.utils.make_grid(output.detach().cpu())
    output = torchvision.transforms.ToPILImage()(output)

    new_im = Image.new('RGB', (human.size[0]*6,human.size[1]*3))
    # left most IO
    new_im.paste(human, (human.size[0]*0, 0))
    new_im.paste(clothes, (human.size[0]*0, human.size[1]*1))
    new_im.paste(skeleton_i, (human.size[0]*0, human.size[1]*2))
    # stage 1
    new_im.paste(skeleton_i, (human.size[0]*2, 0))
    new_im.paste(cm_align, (human.size[0]*3, 0))
    new_im.paste(human_parse_masked, (human.size[0]*4, 0))
    new_im.paste(output1, (human.size[0]*5, 0))
    # stage 0
    new_im.paste(skeleton_i, (human.size[0]*2, human.size[1]*1))
    new_im.paste(parsing_c, (human.size[0]*3, human.size[1]*1))
    new_im.paste(c_align, (human.size[0]*4, human.size[1]*1))
    new_im.paste(warped_cloth, (human.size[0]*5, human.size[1]*1))
    # stage 2
    new_im.paste(human_masked_w_BG, (human.size[0]*2,human.size[1]*2))
    new_im.paste(output1, (human.size[0]*3,human.size[1]*2))
    new_im.paste(warped_cloth_for_stage2, (human.size[0]*4,human.size[1]*2))
    new_im.paste(output, (human.size[0]*5,human.size[1]*2))

    new_im.save(img_name)

def imsave_onlyIO(inH, inC, result, img_name):
    inC = torchvision.utils.make_grid(inC.detach().cpu())
    inC = torchvision.transforms.ToPILImage()(inC)
    inH = torchvision.utils.make_grid(inH.detach().cpu())
    inH = torchvision.transforms.ToPILImage()(inH)
    result = torchvision.utils.make_grid(result.detach().cpu())
    result = torchvision.transforms.ToPILImage()(result)

    new_im = Image.new('RGB', (inC.size[0]*3,inC.size[1]))
    new_im.paste(inH, (inC.size[0]*0,0))
    new_im.paste(inC, (inC.size[0]*1,0))
    new_im.paste(result, (inC.size[0]*2,0))

    new_im.save(img_name)

def imsave_onlyIPO(inH, inC, skeleton, result, img_name):
    inC = torchvision.utils.make_grid(inC.detach().cpu())
    inC = torchvision.transforms.ToPILImage()(inC)
    inH = torchvision.utils.make_grid(inH.detach().cpu())
    inH = torchvision.transforms.ToPILImage()(inH)
    skeleton = torchvision.utils.make_grid(skeleton.detach().cpu())
    skeleton = torchvision.transforms.ToPILImage()(skeleton)
    result = torchvision.utils.make_grid(result.detach().cpu())
    result = torchvision.transforms.ToPILImage()(result)#.convert("RGBA")

    # newImage = []
    # for index, item in enumerate(result.getdata()):
    #     if item[0] < 10 and item[1] <10 and item[2] < 10:
    #         newImage.append((0, 0, 0, 0))
    #     else:
    #         newImage.append(item)
    # result.putdata(newImage)

    # result_bg = Image.new('RGB', (inC.size[0],inC.size[1]), (255, 255, 255))
    # result_bg.paste(result, (0,0), result)

    new_im = Image.new('RGB', (inC.size[0]*4,inC.size[1]))#, (0, 0, 255))
    new_im.paste(inH, (inC.size[0]*0,0))
    new_im.paste(inC, (inC.size[0]*1,0))
    new_im.paste(skeleton, (inC.size[0]*2,0))
    new_im.paste(result, (inC.size[0]*3,0))

    new_im.save(img_name)

def imsave_checkWierdLoss(inH, inC, skeleton, result, gt, img_name):
    inC = torchvision.utils.make_grid(inC.detach().cpu())
    inC = torchvision.transforms.ToPILImage()(inC)
    inH = torchvision.utils.make_grid(inH.detach().cpu())
    inH = torchvision.transforms.ToPILImage()(inH)
    skeleton = torchvision.utils.make_grid(skeleton.detach().cpu())
    skeleton = torchvision.transforms.ToPILImage()(skeleton)
    result = torchvision.utils.make_grid(result.detach().cpu())
    result = torchvision.transforms.ToPILImage()(result)
    gt = torchvision.utils.make_grid(gt.detach().cpu())
    gt = torchvision.transforms.ToPILImage()(gt)

    new_im = Image.new('RGB', (inC.size[0],inC.size[1]*5))
    new_im.paste(inH, (inC.size[0]*0,0))
    new_im.paste(inC, (inC.size[0]*0,inC.size[1]*1))
    new_im.paste(skeleton, (inC.size[0]*0,inC.size[1]*2))
    new_im.paste(result, (inC.size[0]*0,inC.size[1]*3))
    new_im.paste(gt, (inC.size[0]*0,inC.size[1]*4))

    new_im.save(img_name)

def imsave_onlyIHO(inH, inH_masked, inC, skeleton, result, img_name):
    inC = torchvision.utils.make_grid(inC.detach().cpu())
    inC = torchvision.transforms.ToPILImage()(inC)
    inH = torchvision.utils.make_grid(inH.detach().cpu())
    inH = torchvision.transforms.ToPILImage()(inH)
    inH_masked = torchvision.utils.make_grid(inH_masked.detach().cpu())
    inH_masked = torchvision.transforms.ToPILImage()(inH_masked)
    skeleton = torchvision.utils.make_grid(skeleton.detach().cpu())
    skeleton = torchvision.transforms.ToPILImage()(skeleton)
    result = torchvision.utils.make_grid(result.detach().cpu())
    result = torchvision.transforms.ToPILImage()(result)

    new_im = Image.new('RGB', (inC.size[0]*5,inC.size[1]))
    new_im.paste(inH, (inC.size[0]*0,0))
    new_im.paste(inH_masked, (inC.size[0]*1,0))
    new_im.paste(inC, (inC.size[0]*2,0))
    new_im.paste(skeleton, (inC.size[0]*3,0))
    new_im.paste(result, (inC.size[0]*4,0))

    new_im.save(img_name)

def imsave_trainProcess(imgs, img_name, shape=(1024, 768)):
    img_num = len(imgs)
    num_per_row = min(len(imgs), 5)
    col = num_per_row
    row = img_num // num_per_row if img_num % num_per_row == 0 else img_num // num_per_row + 1
    new_im = Image.new('RGB', (shape[1]*num_per_row,shape[0]*row))
    for idx, img in enumerate(imgs):
        col = idx % num_per_row
        row = idx // num_per_row
        img = torchvision.utils.make_grid(img.detach().cpu())
        img = torchvision.transforms.ToPILImage()(img)
        new_im.paste(img, (shape[1]*col, shape[0]*row))
    new_im.save(img_name)

def gen_bgmask2(gen_img, shape=(1024, 768)):#, warped_cloth_for_stage2) :
    img_array = gen_img.detach().cpu().permute(1, 2, 0).numpy()
    img_array = cv2.normalize(img_array,  None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img2gray = cv2.cvtColor(img_array.copy(),cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 0, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    bg_ = ~mask / 255

    bg_mask = torch.zeros((1, shape[0], shape[1]), dtype=torch.float32)
    bg_mask[0] = torch.from_numpy(bg_).type(torch.cuda.FloatTensor) #[0,1]        
    bg_mask.unsqueeze_(0)

    return bg_mask.to(device="cuda")