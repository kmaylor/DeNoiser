import numpy as np
import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
import pickle as pk
from os.path import exists
from os import makedirs
import types
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv1D, Conv2D, Conv3D,Cropping3D, Cropping2D, Cropping1D, UpSampling1D, UpSampling3D
from keras.layers import LeakyReLU, Dropout, Lambda
from keras.optimizers import Adam
from keras.backend import log, count_params, int_shape
from keras.initializers import TruncatedNormal
from keras.utils import multi_gpu_model
import tensorflow as tf
from keras import backend as K
import os

class DeNoiser(object):
    
    def __init__(self, dimensions,
                       channel=1,
                       min_features=64,
                       drop_rate=[.4,.4],
                       kernels = [6,6,6,4,4],
                       strides = [2,2,2,2,2],
                       feature_scale = None,
                       ):
        """
        dimensions: list of the size of each dimension of the data
        channel: number of channels
        min_features: minimum number of features to use in the first layer of the encoder
                      increases according to feature_scale with each layer
        drop_rate: 2 element list containing the drop rate for the encoder and decoder
        kernels: size of the kernels for each layer, also the length of the shorter of  
                 kernels or strides determines the number of layers
        strides: amount by which to scale features, also the length of the shorter of  
                 kernels or strides determines the number of layers
        feature_scale: function or list which determines how to scale the number of 
                       features in each layer, default is a factor of two
        """
        self.dims = dimensions
        self.channel = channel
        self.E = None   # encoder
        self.D = None   # decoder
        self.ED = None  # denoiser model
        self.n_features = min_features
        self.e_drop_r = drop_rate[0]
        self.d_drop_r = drop_rate[1]
        self.kernels = kernels
        self.strides = strides
        #Set scaling for features
        if feature_scale == None:
            feature_scale = lambda:((2*np.ones(len(self.kernels)))**np.arange(len(self.kernels))).astype('int')
            self.feature_scale = feature_scale()
        elif isinstance(self.depth_scale,types.FunctionType):
            self.feature_scale = feature_scale()
        else:
            self.feature_scale = feature_scale
        #Set dimensions of upscaling and convolution 
        self.Conv = self.set_convolution()
        self.UpS = self.set_upsample()
        self.Cropping = self.set_cropping()
    
    def set_convolution(self):
        d = len(self.dims)
        if d == 1:
            return Conv1D
        elif d == 2:
            return Conv2D
        else:
            return Conv3D
            
    def set_upsample(self):
        d = len(self.dims)
        if d == 1:
            return UpSampling1D
        elif d == 2:
            return self.UpSampling2DBilinear
        else:
            return UpSampling3D
    
    def set_cropping(self):
        d = len(self.dims)
        if d == 1:
            return Cropping1D
        elif d == 2:
            return Cropping2D
        else:
            return Cropping3D
    
    def encoder(self):
        if self.E:
            return self.E
        self.E = Sequential(name='encoder')
        n_features = self.n_features
        feature_scale = self.feature_scale
        dropout = self.e_drop_r
        input_shape = list(self.dims)+[self.channel]
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
        dropout = self.d_drop_r
        input_shape = self.encoder().layers[-1].output_shape[1:]
        self.D.add(self.Conv(n_features*feature_scale[0], self.kernels[-1], strides=1, input_shape=input_shape,\
            padding='same',name='D_con_1'))
        self.D.add(self.UpS(self.strides[-1],name='UpS_1'))
        self.D.add(LeakyReLU(alpha=0.2,name='leak_1'))
        for i,ks in enumerate(zip(self.kernels[::-1][1:],self.strides[::-1][1:])):
            if i+1 == len(self.kernels): break
            self.D.add(self.Conv(n_features*feature_scale[i+1], ks[0], strides=1,    padding='same',name='D_con_%i'%(i+2)))
            self.D.add(self.UpS(ks[1],name='UpS_%i'%(i+2)))
            self.D.add(LeakyReLU(alpha=0.2,name='leak_%i'%(i+2)))
            self.D.add(Dropout(dropout,name='drop_%i'%(i+2)))
        self.D.add(self.Conv(1, self.kernels[-1], strides=1, padding='same',name='D_con_%i'%(i+3)))
        self.D.add(Activation('tanh', name = 'Tanh'))
        diff = lambda i,d: int((self.D.get_layer('Tanh').output_shape[i+1]-d)/2)
        crop_size = [(diff(i,d),diff(i,d)) for i,d in enumerate(self.dims)] 
        if len(self.dims)==1: crop_size=crop_size[0]
        self.D.add(self.Cropping(cropping=crop_size, name='Crop_%iD'%(len(self.dims))))
        self.D.summary()
        return self.D
    
    
    
    def denoiser_model(self):
        if self.ED:
            return self.ED
        optimizer = Adam(lr=0.002,beta_1=0.9, decay=0)
        self.ED = Sequential(name='EDmodel')
        self.ED.add(self.encoder())
        self.ED.add(self.decoder())
        self.ED.compile(loss='mean_squared_error', optimizer=optimizer,\
            metrics=['accuracy'])
        self.ED.summary()
        return self.ED
        
    def UpSampling2DBilinear(self,stride, **kwargs):
            def layer(x):
                input_shape = K.int_shape(x)
                output_shape = (stride * input_shape[1], stride * input_shape[2])
                return K.tf.image.resize_images(x, output_shape, align_corners=True,
                        method = K.tf.image.ResizeMethod.BILINEAR)
            return Lambda(layer, **kwargs)