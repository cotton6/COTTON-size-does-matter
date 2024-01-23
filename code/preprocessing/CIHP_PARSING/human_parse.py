import numpy as np
import os
from PIL import Image
import cv2
import sys
from datetime import datetime
import time
import torch
import torchvision
import numpy as np
from PIL import Image
import argparse
import glob
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

IMG_MEAN = np.array((125.0, 114.4, 107.9), dtype=np.float32)
N_CLASSES = 20

label_colours_20 = [(0,0,0)
            , (128,0,0), (255,0,0), (0,85,0), (170,0,51), (255,85,0)
            ,(0,0,85), (0,119,221), (85,85,0), (0,85,85), (85,51,0)
            ,(52,86,128), (0,128,0), (0,0,255), (51,170,221), (0,255,255)
            ,(85,255,170), (170,255,85), (255,255,0), (255,170,0)]

label_colours_15 = [(0,0,0)
    # 0=Background
    ,(255,0,0),(0,0,255),(85,51,0),(255,85,0),(0,119,221)
    # 1=Hat,  2=Hair,    3=Glove, 4=Sunglasses, 5=UpperClothes
    ,(0,0,85),(0,85,85),(51,170,221),(0,255,255),(85,255,170)
    # 6=Dress, 7=Coat, 8=Socks, 9=Pants, 10=Jumpsuits/Neck
    ,(170,255,85),(255,255,0),(255,170,0),(85,85,0),(0,255,255)
    # 11=Scarf, 12=Skirt, 13=Face, 14=LeftArm, 15=RightArm
    ,(85,255,170),(170,255,85),(255,255,0),(255,170,0)]
    # 16=LeftLeg, 17=RightLeg, 18=LeftShoe, 19=RightShoe
    
Frozen_model_path = "checkpoint/CIHP_pgn/frozen_inference_graph_GPU.pb"

def read_labeled_image_list(data_dir, data_list):
    """Reads txt file containing paths to images and ground truth masks.
    
    Args:
      data_dir: path to the directory with images and masks.
      data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.
       
    Returns:
      Two lists with all file names for images and masks, respectively.
    """
    images = []
    for image_name in data_list:
        images.append(os.path.join(data_dir, image_name))

    return images

def read_images_from_disk(input_queue, input_size, random_scale, random_mirror=False): # optional pre-processing arguments
    """Read one image and its corresponding mask with optional pre-processing.
    
    Args:
      input_queue: tf queue with paths to the image and its mask.
      input_size: a tuple with (height, width) values.
                  If not given, return images of original size.
      random_scale: whether to randomly scale the images prior
                    to random crop.
      random_mirror: whether to randomly mirror the images prior
                    to random crop.
      
    Returns:
      Two tensors: the decoded image and its mask.
    """
    # print("input_queue.size = ", len(input_queue))
    # print("======= input_queue[0] = ", input_queue[0])
    img_contents = tf.read_file(input_queue[0])
    #img_contents = tf.read_file(input_queue[0])
    #label_contents = tf.read_file(input_queue[1])
    #edge_contents = tf.read_file(input_queue[2])

    
    img = tf.image.decode_jpeg(img_contents, channels=3)
    img_r, img_g, img_b = tf.split(value=img, num_or_size_splits=3, axis=2)
    img = tf.cast(tf.concat([img_b, img_g, img_r], 2), dtype=tf.float32)
    # Extract mean.
    img -= IMG_MEAN

    if input_size is not None:
        h, w = input_size
        # new_shape = tf.squeeze(tf.stack([h, w]), squeeze_dims=[1])
        img = tf.image.resize_images(img, [w, h], method=tf.image.ResizeMethod.AREA)#, method='area')
        print("INPUT RESIZE | img.shape = {}".format(img.shape))
        # Randomly scale the images and labels.
        # if random_scale:
        #     #img, label, edge = image_scaling(img, label, edge)
        #     img = image_scaling(img)

        # # Randomly mirror the images and labels.
        # if random_mirror:
        #     #img, label, edge = image_mirroring(img, label, edge)
        #     img = image_mirroring(img, None, None)

        # # Randomly crops the images and labels.
        # #img, label, edge = random_crop_and_pad_image_and_labels(img, label, edge, h, w, IGNORE_LABEL)
        # img = random_crop_and_pad_image_and_labels(img, None, None, h, w, IGNORE_LABEL)

    return img

def ImageReader(data_dir, data_list, data_id_list, input_size, random_scale,
                 random_mirror, shuffle):
    '''Initialise an ImageReader.
    
    Args:
        data_dir: path to the directory with images and masks.
        data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.
        data_id_list: path to the file of image id.
        input_size: a tuple with (height, width) values, to which all the images will be resized.
        random_scale: whether to randomly scale the images prior to random crop.
        random_mirror: whether to randomly mirror the images prior to random crop.
        coord: TensorFlow queue coordinator.
    '''

    #self.image_list, self.label_list = read_labeled_image_list(self.data_dir, self.data_list)
    image_list = read_labeled_image_list(data_dir, data_list)
    # print("\n\n\nself.image_list = ", image_list)
    images = tf.convert_to_tensor(image_list, dtype=tf.string)
    queue = tf.train.slice_input_producer([images], shuffle=shuffle) 
    image = read_images_from_disk(queue, input_size, random_scale, random_mirror)
    return image, image_list

def parsing_init(DATA_DIR, parsing_graph, sess):
    data_list = os.listdir(DATA_DIR)
    NUM_STEPS = len(data_list)

    # Load reader.
    with parsing_graph.as_default():
        with tf.name_scope("create_inputs"):
            image, image_list = ImageReader(DATA_DIR, data_list, None, None, False, False, False)
            image_rev = tf.reverse(image, tf.stack([1]))

        image_batch = tf.stack([image, image_rev])

    image_tensor_detect = parsing_graph.get_tensor_by_name('stack:0')
    pred_all = parsing_graph.get_tensor_by_name('ExpandDims_1:0')
    
    return image_tensor_detect, image_batch, pred_all, data_list

def parsing_init_():
    # Print GPUs available
    print(tf.config.list_physical_devices('GPU'))

    # Adjusting percentage of GPU available for tensorflow
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)#0.2)
    config = tf.ConfigProto(gpu_options=gpu_options)

    # Build graph with pre-saved graph and GPU settings
    parsing_graph = tf.Graph()
    with parsing_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(Frozen_model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            _ = tf.import_graph_def(od_graph_def, name='')
        parsing_sess = tf.Session(graph=parsing_graph, config=config)
        # Initialization
        init = tf.global_variables_initializer()
        parsing_sess.run(init)
    return parsing_sess, parsing_graph

def convert_upper_label(human1_parse_img):
    # TODO: upper
    # labels = {
    #             0:  ['background',      [0]],
    #             1:  ['hair',            [1, 2]],
    #             2:  ['Face',            [4, 13]],
    #             3:  ['Neck',            [10]],
    #             4:  ['Upper_clothes',   [5]],
    #             5:  ['Coat',            [7]],
    #             6:  ['Dress',           [6]],
    #             7:  ['Lower_clothes',   [9, 12]],
    #             8:  ['Left_arm',        [14]],
    #             9:  ['Right_arm',       [15]],
    #             10:  ['Left_leg',       [16]],
    #             11:  ['Right_leg',      [17]],
    #             12:  ['Left_shoe',      [18]],
    #             13:  ['Right_shoe',     [19]],
    #             14:  ['Accessories',    [3, 8, 11]]
    #         }

    bg              = (human1_parse_img == 0).astype(np.float32)
    hair            = (human1_parse_img == 1).astype(np.float32) + (human1_parse_img == 2).astype(np.float32)
    Face            = (human1_parse_img == 4).astype(np.float32) + (human1_parse_img == 13).astype(np.float32)
    Neck            = (human1_parse_img == 10).astype(np.float32)
    Upper_clothes   = (human1_parse_img == 5).astype(np.float32)
    Coat            = (human1_parse_img == 7).astype(np.float32)
    Dress           = (human1_parse_img == 6).astype(np.float32)
    Lower_clothes   = (human1_parse_img == 9).astype(np.float32) + (human1_parse_img == 12).astype(np.float32)
    Left_arm        = (human1_parse_img == 14).astype(np.float32)
    Right_arm       = (human1_parse_img == 15).astype(np.float32)
    Left_leg        = (human1_parse_img == 16).astype(np.float32)
    Right_leg       = (human1_parse_img == 17).astype(np.float32)
    Left_shoe       = (human1_parse_img == 18).astype(np.float32)
    Right_shoe      = (human1_parse_img == 19).astype(np.float32)
    Accessories     = (human1_parse_img == 3).astype(np.float32) + (human1_parse_img == 8).astype(np.float32)\
                        + (human1_parse_img == 11).astype(np.float32)

    merge_channel = 15
    parsing_height, parsing_width = human1_parse_img.shape
    merge_parse = torch.zeros((merge_channel, parsing_height, parsing_width), dtype=torch.float32)
    merge_parse[0] = torch.from_numpy((bg-0.5)*2).type(torch.cuda.FloatTensor) #[0,1]->[-1,1]                    
    merge_parse[1] = torch.from_numpy((hair-0.5)*2).type(torch.cuda.FloatTensor) #[0,1]->[-1,1]
    merge_parse[2] = torch.from_numpy((Face-0.5)*2).type(torch.cuda.FloatTensor) #[0,1]->[-1,1]
    merge_parse[3] = torch.from_numpy((Neck-0.5)*2).type(torch.cuda.FloatTensor) #[0,1]->[-1,1]
    merge_parse[4] = torch.from_numpy((Upper_clothes-0.5)*2).type(torch.cuda.FloatTensor) #[0,1]->[-1,1]
    merge_parse[5] = torch.from_numpy((Coat-0.5)*2).type(torch.cuda.FloatTensor) #[0,1]->[-1,1]                    
    merge_parse[6] = torch.from_numpy((Dress-0.5)*2).type(torch.cuda.FloatTensor) #[0,1]->[-1,1]
    merge_parse[7] = torch.from_numpy((Lower_clothes-0.5)*2).type(torch.cuda.FloatTensor) #[0,1]->[-1,1]
    merge_parse[8] = torch.from_numpy((Left_arm-0.5)*2).type(torch.cuda.FloatTensor) #[0,1]->[-1,1]                    
    merge_parse[9] = torch.from_numpy((Right_arm-0.5)*2).type(torch.cuda.FloatTensor) #[0,1]->[-1,1]
    merge_parse[10] = torch.from_numpy((Left_leg-0.5)*2).type(torch.cuda.FloatTensor) #[0,1]->[-1,1]                 
    merge_parse[11] = torch.from_numpy((Right_leg-0.5)*2).type(torch.cuda.FloatTensor) #[0,1]->[-1,1]
    merge_parse[12] = torch.from_numpy((Left_shoe-0.5)*2).type(torch.cuda.FloatTensor) #[0,1]->[-1,1]
    merge_parse[13] = torch.from_numpy((Right_shoe-0.5)*2).type(torch.cuda.FloatTensor) #[0,1]->[-1,1]
    merge_parse[14] = torch.from_numpy((Accessories-0.5)*2).type(torch.cuda.FloatTensor) #[0,1]->[-1,1]

    merge_parse.unsqueeze_(0)

    return merge_parse

def decode_labels(mask, labels, num_classes=15):
    """Decode batch of segmentation masks.
    
    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
    
    Returns:
      A batch with num_images RGB images of the same size as the input. 
    """
    n, h, w = mask.size()
    #print("mask.size()= ", mask.size())
    #assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((n, h, w, 3), dtype=np.uint8)
    #p_label_colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    for i in range(n):
        img = Image.new('RGB', (w, h))
        pixels = img.load()
        for j_, j in enumerate(mask[i, :, :]): #j_=h, j=tensor(1*w)
            for k_, k in enumerate(j): #k_=w, k=tensor (1*1)
                if k < num_classes:
                    pixels[k_,j_] = labels[k] #for whole parsing: label_colours[k]
        outputs[i] = np.array(img)
    outputs = np.transpose(outputs, (0,3,1,2))
    outputs = torch.from_numpy(outputs)
    return outputs

def decode_labels_inside(mask, num_images=1, num_classes=21):
    """Decode batch of segmentation masks.
    
    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).
    
    Returns:
      A batch with num_images RGB images of the same size as the input. 
    """
    n, h, w, c = mask.shape
    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
      img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
      pixels = img.load()
      for j_, j in enumerate(mask[i, :, :, 0]):
          for k_, k in enumerate(j):
              if k < num_classes:
                  pixels[k_,j_] = label_colours_20[k]
      outputs[i] = np.array(img)
    return outputs

def visualize_parse(parse):
    '''
        input:  parsing mask (batch_size, channel, h, w) in pytorch Tensor [-1,1]
        output: parsing image (batch_size, h, w) in pytorch Tensor [0,255] with mode 'RGB'
    '''
    if parse.size()[1] == 20:
        parse = torch.argmax(parse.cpu(), dim=1)
        parse_img = decode_labels(parse, label_colours_20, num_classes=20)
    elif parse.size()[1] == 15:
        parse = torch.argmax(parse.cpu(), dim=1)
        parse_img = decode_labels(parse, label_colours_15, num_classes=15)
    return parse_img

def convert_label_20to15(human_parsing):

    bg              = (human_parsing == 0)
    hair            = (human_parsing == 1) + (human_parsing == 2)
    Face            = (human_parsing == 4) + (human_parsing == 13)
    Neck            = (human_parsing == 10)
    Upper_clothes   = (human_parsing == 5)
    Coat            = (human_parsing == 7)
    Dress           = (human_parsing == 6)
    Lower_clothes   = (human_parsing == 9) + (human_parsing == 12)
    Left_arm        = (human_parsing == 14)
    Right_arm       = (human_parsing == 15)
    Left_leg        = (human_parsing == 16)
    Right_leg       = (human_parsing == 17)
    Left_shoe       = (human_parsing == 18)
    Right_shoe      = (human_parsing == 19)
    Accessories     = (human_parsing == 3) + (human_parsing == 8) + (human_parsing == 11)

    img = np.zeros_like(human_parsing)
    img[bg            ] = 0
    img[hair          ] = 1
    img[Face          ] = 2
    img[Neck          ] = 3
    img[Upper_clothes ] = 4
    img[Coat          ] = 5
    img[Dress         ] = 6
    img[Lower_clothes ] = 7
    img[Left_arm      ] = 8
    img[Right_arm     ] = 9
    img[Left_leg      ] = 10
    img[Right_leg     ] = 11
    img[Left_shoe     ] = 12
    img[Right_shoe    ] = 13
    img[Accessories   ] = 14

    return img

def parsing_gen(input_dir, output_dir):
    parsing_sess, parsing_graph = parsing_init_()
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "vis"), exist_ok=True)

    image_tensor_detect, image_batch, pred_all, data_list = parsing_init(input_dir, parsing_graph, parsing_sess)
    
    # For multi-thread processing
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=parsing_sess)

    for step in range(len(data_list)):
        start_time = time.time()
        image_numpy = image_batch.eval(session=parsing_sess) 
        parsing_= parsing_sess.run(pred_all,feed_dict={image_tensor_detect: image_numpy})

        msk = decode_labels_inside(parsing_, num_classes=N_CLASSES)
        parsing_im = Image.fromarray(msk[0])
        file_id = data_list[step][:-4]
        parsing_im.save('{}/vis/{}.png'.format(output_dir, file_id))
        cv2.imwrite('{}/{}.png'.format(output_dir, file_id), parsing_[0,:,:,0])
        
        if step == 0:
            print("step {} | cost {} sec".format(step, time.time()-start_time))
        else:
            print("\r step [{}/{}] | cost {} sec".format(step, len(data_list), time.time()-start_time), end=" ")
        
    # Stop all thread, ready to finish
    coord.request_stop()
    coord.join(threads)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default='../pose_filtered_Data', type=str, help='Name of category')
    parser.add_argument("--brand", default='Example_top', type=str, help='Name of category')
    parser.add_argument("--cat", default=None, type=str, help='Name of category')
    opt = parser.parse_args()
    print(opt)
    cat_list = [opt.cat] if opt.cat is not None else [os.path.basename(cat) for cat in glob.glob(os.path.join(opt.root, opt.brand, '*'))]
    print(cat_list)
    for cat in cat_list:
        print('='*10 + " {} -- {} ".format(opt.brand, cat) + "="*20)
        parsing_gen(os.path.join(opt.root, opt.brand, cat, 'model'), os.path.join(opt.root, opt.brand, cat, 'CIHP'))