import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle as pk
from os.path import exists
from os import makedirs
import types
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv1D, Conv2D, Cropping2D
from keras.layers import LeakyReLU, Dropout, Lambda
from keras.optimizers import Adam
from keras.backend import log, count_params, int_shape
from keras.initializers import TruncatedNormal
from keras.utils import multi_gpu_model
import tensorflow as tf
from keras import backend as K
import os

class DeNoiser(object):
    
    def __init__(self, dims,
                       channel=1,
                       min_features=64,
                       drop_rate=[.4,.4],
                       kernels = [6,4,2],
                       strides = [2,2,2],
                       feature_scale = None,
                       ):

        self.dims = dims
        self.channel = channel
        self.E = None   # encoder
        self.D = None   # decoder
        self.DN = None  # denoiser model
        self.n_feautures = max_features
        self.e_drop_r = drop_rate[0]
        self.d_drop_r = drop_rate[1]
        self.kernels = kernels
        self.strides = strides
        if feature_scale == None:
            feature_scale = lambda:((2*np.ones(len(self.kernels)))**np.arange(len(self.kernels))).astype('int')
            self.feature_scale = feature_scale()
        elseif isinstance(self.depth_scale,types.FunctionType):
            self.feature_scale = feature_scale()
        else:
            self.feature_scale = feature_scale
        self.Conv = set_convolution(dims)
        self.UpS = set_upsample(dims)
    
    def set_convolution(dims):
        d = len(dims)
        if d == 1:
            return Conv1D
        elif d == 2:
            return Conv2D
            
    def set_upsample(dims):
        d = len(dims)
        if d == 1:
            return need to do
        elif d == 2:
            return self.UpSampling2DBilinear
    
    def encoder(self):
        if self.E:
            return self.E
        self.E = Sequential(name='encoder')
        n_features = self.n_features
        feature_scale = self.feature_scale
        dropout = self.edrop_r
        input_shape = self.dims.extend(self.channel)
        self.E.add(self.Conv(n_features*feature_scale[0], self.kernels[0], strides=self.strides[1], input_shape=input_shape,\
                padding='same',name='E_con_1'))
        self.E.add(LeakyReLU(alpha=0.2,name='leak_1'))
        self.E.add(Dropout(dropout,name='drop_1'))
        for i,ks in enumerate(zip(self.kernels[1:],self.strides[1:])):
            self.E.add(self.Conv(n_features*feature_scale[i+1], ks[0], strides=ks[1],\
                padding='same',name='con_%i'%(i+2)))
            self.E.add(LeakyReLU(alpha=0.2,name='leak_%i'%(i+2)))
            self.E.add(Dropout(dropout,name='drop_%i'%(i+2)))
        self.E.summary()
        return self.E
    
    def decoder(self):
        if self.D:
            return self.D
        self.D = Sequential(name='decoder')
        n_features = self.n_features
        feature_scale = self.feature_scale[::-1]
        dropout = self.ddrop_r
        input_shape = (self.length, self.channel)
        upsample_layer = self.UpSampling2DBilinear
        self.D.add(self.Conv(features*feature_scale[0], self.kernels[-1], strides=1, input_shape=input_shape,\
                padding='same',name='D_con_1'))
        self.D.add(self.UpS(self.strides[-1],name='UpS_1'))
        self.D.add(LeakyReLU(alpha=0.2,name='leak_1'))
        for i,ks in enumerate(zip(self.kernels[::-1][1:],self.strides[::-1][1:])):
            self.D.add(self.Conv(features*feature_scale[i+1], ks[0], strides=1, input_shape=input_shape,\
                padding='same',name='con_%i'%(i+2)))
            self.D.add(self.UpS(ks[1],name='UpS_%i'%(i+2))))
            if i+1 == len(self.kernels):
                self.D.add(Activation('tanh', name = 'Tanh'))
            else:
                self.D.add(LeakyReLU(alpha=0.2,name='leak_%i'%(i+2)))
            self.D.add(Dropout(dropout,name='drop_%i'%(i+2)))
        self.D.summary()
        return self.D
    
    def denoiser(self):
        if self.AE:
            return self.AE
        optimizer = Adam(lr=0.0002,beta_1=0.5, decay=0)
        self.AE = Sequential(name='aemodel')
        self.AE.add(self.encoder())
        self.AE.add(self.decoder())
        self.AE.compile(loss='mean_squared_error', optimizer=optimizer,\
            metrics=['accuracy'])
        self.AE.summary()
        return self.AE
        
    def UpSampling2DBilinear(self,stride, **kwargs):
            def layer(x):
                input_shape = K.int_shape(x)
                output_shape = (stride * input_shape[1], stride * input_shape[2])
                return K.tf.image.resize_images(x, output_shape, align_corners=True,
                        method = K.tf.image.ResizeMethod.BILINEAR)
            return Lambda(layer, **kwargs)