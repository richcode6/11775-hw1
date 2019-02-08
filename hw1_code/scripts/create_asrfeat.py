#!/bin/python
import numpy as np
import os
import pickle
import sys
import string
import math


def create_vector(file, vocab):
    v = np.zeros(len(vocab))
    non_zero = False
    for line in file:
        line = line.lower().replace('\n', '').replace('.',' ').split()
        # print(line)
        for word in line:
            word = word.strip()
            if word in vocab:
               non_zero = True
               v[vocab[word]] += 1
    # print(v)
    return v, non_zero


def generate_vocab(file, size):
    v = {}
    df = {}
    doc = 0
    for line in file:
        line = line.replace('\n', '').split()
        id = 'asr/'+line[0]+'.txt'
        if not os.path.exists(id):
            continue
        # print('hello')
        asr_file = open(id, 'r')
        doc += 1
        for line in asr_file:
            line = line.lower().replace('\n', '').replace('.','').replace(',','').split()
            f = {}
            for word in line:
                word = word.strip()
                if word in v:
                    v[word] += 1
                else:
                    v[word] = 1
                if word not in f:
                   if word in df:
                      df[word] += 1
                   else:
                      df[word] = 1  
                   f[word] = 1

    vocab = {}
    keys = sorted(v, key=v.get, reverse=True)
    print("Total unique words in training: {}".format(len(keys)))
    size = min(size, len(keys))
    print("Size of Vocabulary: " + str(size))
    k = 0
    v = {w: v[w]*math.log(doc/df[w]) for w in v}
    keys = sorted(v, key=v.get, reverse=True)
    with open('feat/vocab_{}.txt'.format(size), 'w') as f:
        for i in range(len(keys)):
            f.write(keys[i]+'\n')
            if v[keys[i]] in range(10, 550) and "'" not in keys[i] and len(keys[i])> 2:
                vocab[keys[i]] = k
                k += 1
            if k == size:
                break
    print("Vocabulary generated successfully ! -> feat/vocab_{}.txt".format(size))
    return vocab


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: {0} path/to/train_list path/to/all_list vocab_len".format(sys.argv[0]))
        print("train_list -- the list of videos in training set")
        print("all_list -- the list of all videos (train + val + test)")
        print("vocab_len -- the size of vocabulary")
        exit(1)

    trn_vid = open(sys.argv[1], 'r').readlines()
    vocab = generate_vocab(trn_vid, int(sys.argv[2]))
    print("Generating ASR features using TF-IDF")
    model_feat = {}
    df = np.zeros(len(vocab))
    total_frames = 0
    for line in open(sys.argv[1], 'r').readlines():
        line = line.replace('\n', '').split()
        asr_file = 'asr/'+line[0]+'.txt'
        if not os.path.exists(asr_file):
            continue
        model_feat[line[0]], non_zero = create_vector(open(asr_file, 'r'), vocab)
        # normalize to get tf
        if non_zero:
            model_feat[line[0]] = np.divide(model_feat[line[0]], np.sum(model_feat[line[0]]))
        df += model_feat[line[0]]
        total_frames += 1

    for id in model_feat:
       model_feat[id] *= np.log(total_frames / df)

    with open('feat/asr_dict_{}.pickle'.format(len(vocab)), 'wb') as w:
        pickle.dump(model_feat, w)

    print("ASR features generated successfully! -> {}".format('feat/asr_dict_{}.pickle'.format(len(vocab))))
