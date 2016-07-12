import caffe

import numpy as np
from PIL import Image

import random


class SegDataLayer(caffe.Layer):
    """
    Load (input image, label image) pairs from database, 
    like PASCAL VOC, CityScape etc.

    Use this to feed data to a segmentation network
    """

    def setup(self, bottom, top):
        """
        parameters:

        - imagelist
        - labellist
        - mean
        - randomize (True)
        - seed (None)
        
        """
        # config
        params = eval(self.param_str)
        self.imagelist = params['imagelist']
        self.labellist = params['labellist']
        self.mean = np.array(params['mean'])
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)
        self.scale = params.get('scale', 4)

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")
        
        # load address of images and labels
        self.imageindices = open(self.imagelist, 'r').read().splitlines()
        self.labelindices = open(self.labellist, 'r').read().splitlines()
        self.idx = 0
        if len(self.imageindices) != len(self.labelindices):
            raise Exception("Images and labels should pair.")

        # random
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.imageindices)-1)


    def reshape(self, bottom, top):
        # load image + label pair
        self.data = self.load_image(self.idx)
        self.label = self.load_label(self.idx)
        # reshape tops to fit
        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.label.shape)


    def forward(self, bottom, top):
        # assign outpout
        top[0].data[...] = self.data
        top[1].data[...] = self.label

        # pick next input
        if self.random:
            self.idx = random.randint(0, len(self.imageindices)-1)
        else:
            self.idx += 1
            if self.idx == len(self.indices):
                self.idx = 0


    def backward(self, top, propagate_down, bottom):
        pass


    def load_image(self, idx):
        im = Image.open(self.imageindices[idx])
        if self.scale != 1:
            im = im.resize(im.size/np.array(self.scale, np.int))
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:,:,::-1]
        in_ -= self.mean
        in_ = in_.transpose((2,0,1))
        return in_

    def load_label(self, idx):
        im = Image.open(self.labelindices[idx])
        if self.scale != 1:
            im = im.resize(im.size/np.array(self.scale, np.int))
        label = np.array(im, dtype=np.uint8)
        label = label[np.newaxis, ...]
        return label

