#!/usr/bin/env python3

import os
import sys
import threading
import cv2
import numpy as np
import yaml
import pickle
import pdb
# import zip
from PIL import Image
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data

from torchvision.models import resnet50
from torchvision.models import resnet18
from torchvision import transforms

import threading
import time
import multiprocessing


def get_cnn_features_from_video(downsampled_video_filename, cnn_feat_video_filename, keyframe_interval, idx):
    # Receives filename of downsampled video and of output path for features.
    # Extracts features in the given keyframe_interval. Saves features in pickled file.

    final = []
    bidx = 0
    start = time.time()
    for batch_frame in get_keyframes(downsampled_video_filename, keyframe_interval):
        x = Variable(batch_frame)
        if use_resnet50:
            z = cnn(x)
            z = z.view(z.size(0), -1).data.numpy()
        if use_resnet18:
            my_embedding[idx] = torch.zeros(len(batch_frame), 512, 1, 1)

            def copy_data(m, i, o):
                my_embedding[idx].copy_(o.data)

            h = layer.register_forward_hook(copy_data)
            h_x = model(x)
            h.remove()
            z = my_embedding[idx].data.numpy()
            z = z.squeeze(-1).squeeze(-1)

        bidx += 1
        # print('{} batches completed.'.format(bidx))
        final.append(z)

    end = time.time() - start
    final = np.vstack(final)
    print(final.shape, end/60)
    np.savez(cnn_feat_video_filename, final)
    # pickle.dump(kp, open(surf_feat_video_filename, "wb"))


def get_keyframes(downsampled_video_filename, keyframe_interval):
    "Generator function which returns the next keyframe."

    # Create video capture object
    video_cap = cv2.VideoCapture(downsampled_video_filename)
    frame = 0
    bsz = 1
    batch = []
    while True:
        frame += 1
        ret, img = video_cap.read()
        if ret is False:
            break
        if img is None:
            continue
        img = Image.fromarray(img).convert('RGB')
        img = transform(img)
        if frame % keyframe_interval == 0:
            batch.append(img)
            if len(batch) == bsz:
                x = torch.stack(batch)
                batch = []
                yield x

    if len(batch) > 0:
        print(len(batch))
        x = torch.stack(batch)
        yield x
    video_cap.release()


def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def get_cnn_features_from_videos(fread, keyframe_interval, idx):
    i = 0
    for line in fread:
        video_name = line.replace('\n', '')
        downsampled_video_filename = os.path.join(downsampled_videos, video_name + '.ds.mp4')
        cnn_feat_video_filename = os.path.join(cnn_features_folderpath, video_name + '.cnn')

        if not os.path.isfile(downsampled_video_filename):
            continue
        if os.path.exists(cnn_feat_video_filename+".npz"):
            continue
        # Get SURF features for one video
        get_cnn_features_from_video(downsampled_video_filename,
                                     cnn_feat_video_filename, keyframe_interval, idx)
        i += 1
        if i%500==0:
            print("Processed {}".format(i))


if __name__ == '__main__':
    print(sys.argv)
    if len(sys.argv) != 3:
        print("Usage: {0} video_list config_file".format(sys.argv[0]))
        print("video_list -- file containing video names")
        print("config_file -- yaml filepath containing all parameters")
        exit(1)

    all_video_names = sys.argv[1]
    config_file = sys.argv[2]
    my_params = yaml.load(open(config_file))

    # Get parameters from config file
    keyframe_interval = my_params.get('keyframe_interval')
    cnn_features_folderpath = my_params.get('cnn_features')
    downsampled_videos = my_params.get('downsampled_videos')

    # TODO: Create SURF object

    # Check if folder for SURF features exists
    if not os.path.exists(cnn_features_folderpath):
        os.mkdir(cnn_features_folderpath)

    # Loop over all videos (training, val, testing)
    # TODO: get SURF features for all videos but only from keyframes

    fread = open(all_video_names, "r").readlines()
    lines = chunkIt(fread, 1)
    i = 0

    _transforms = list()
    _transforms.append(transforms.Resize((224, 224)))
    _transforms.append(transforms.ToTensor())
    _transforms.append(
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]))
    transform = transforms.Compose(_transforms)

    use_resnet18 = True
    use_resnet50 = False

    if use_resnet18:
        model = resnet18(pretrained=True)
        layer = model._modules.get('avgpool')

    if use_resnet50:
        model = resnet50(pretrained=True)
        cnn = nn.Sequential(*list(model.children())[:-2])
        for param in cnn.parameters():
            param.requires_grad = False
        cnn.train(False)

    # cnn.cuda()
    print("Initialized!")
    # Remove final classifier layer

    thread = [None for _ in range(len(lines))]
    my_embedding = [None for _ in range(len(lines))]

    for i in range(0, len(thread)):
        # p = multiprocessing.Process(target=get_cnn_features_from_videos,
        #                            args=(lines[i], keyframe_interval, i))
        # p.start()

        thread[i] = threading.Thread(target=get_cnn_features_from_videos,
                                   args=(lines[i], keyframe_interval, i))

    for i in range(0, len(thread)):
        thread[i].start()

    for i in range(0, len(thread)):
        thread[i].join()
