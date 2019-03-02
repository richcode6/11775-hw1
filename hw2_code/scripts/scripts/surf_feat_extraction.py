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
import threading


def get_surf_features_from_video(downsampled_video_filename, surf_feat_video_filename, keyframe_interval, hessian_threshold):
    # Receives filename of downsampled video and of output path for features.
    # Extracts features in the given keyframe_interval. Saves features in pickled file.
    kp = None
    k = 0
    d = 0
    surf = cv2.xfeatures2d.SURF_create(hessian_threshold)
    for frame in get_keyframes(downsampled_video_filename, keyframe_interval):
        key, desc = surf.detectAndCompute(frame, None)
        # d += len(desc)
        if kp is None:
            kp = desc
        elif desc is not None:
            kp = np.concatenate((kp, desc))
        k += 1

    print("Keyframes {} {}".format(k, d/k))
    np.savez(surf_feat_video_filename, kp)
    # pickle.dump(kp, open(surf_feat_video_filename, "wb"))


def get_keyframes(downsampled_video_filename, keyframe_interval):
    "Generator function which returns the next keyframe."

    # Create video capture object
    video_cap = cv2.VideoCapture(downsampled_video_filename)
    frame = 0
    while True:
        frame += 1
        ret, img = video_cap.read()
        if ret is False:
            break
        if frame % keyframe_interval == 0:
            yield img
    video_cap.release()


def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def get_surf_features_from_videos(fread, keyframe_interval, hessian_threshold):
    i = 0
    for line in fread:
        video_name = line.replace('\n', '')
        downsampled_video_filename = os.path.join(downsampled_videos, video_name + '.ds.mp4')
        surf_feat_video_filename = os.path.join(surf_features_folderpath, video_name + '.surf')

        if not os.path.isfile(downsampled_video_filename):
            continue
        if os.path.exists(surf_feat_video_filename+".npz"):
            continue
        # Get SURF features for one video
        get_surf_features_from_video(downsampled_video_filename,
                                     surf_feat_video_filename, keyframe_interval, hessian_threshold)
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
    hessian_threshold = my_params.get('hessian_threshold')
    surf_features_folderpath = my_params.get('surf_features')
    downsampled_videos = my_params.get('downsampled_videos')

    # TODO: Create SURF object

    # Check if folder for SURF features exists
    if not os.path.exists(surf_features_folderpath):
        os.mkdir(surf_features_folderpath)

    # Loop over all videos (training, val, testing)
    # TODO: get SURF features for all videos but only from keyframes

    fread = open(all_video_names, "r").readlines()
    lines = chunkIt(fread, 6)
    i = 0

    thread = [None for _ in range(len(lines))]

    for line in lines:
        thread[i] = threading.Thread(target=get_surf_features_from_videos,
                                   args=(line, keyframe_interval, hessian_threshold))
        i += 1

    for i in range(0, len(thread)):
        thread[i].start()

    for i in range(0, len(thread)):
        thread[i].join()
