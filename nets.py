#!/bin/python

from pycuda import driver
from pycuda import gpuarray
from pycuda import autoinit

import mxnet as mx
from mxnet.gluon import nn

import numpy as np
import os

OUT_DIM = 192
FEATURE_DIM = 128
TILE_SIZE = 96

class LeNetHeadLayer(nn.Block):
    def __init__(self,**kwargs):
        super(LeNetHeadLayer,self).__init__(**kwargs)
        net = nn.Sequential()
        with self.name_scope():
            net.add(nn.Conv2D(channels=20, kernel_size=5, activation='relu'))
            net.add(nn.MaxPool2D(pool_size=2, strides=2))
            net.add(nn.Conv2D(channels=50, kernel_size=3, activation='relu'))
            net.add(nn.MaxPool2D(pool_size=2, strides=2))
            net.add(nn.Flatten())
            net.add(nn.Dense(FEATURE_DIM,activation='relu'))
            self.net = net

    def forward(self,x):
        return self.net(x)
    
class GPUMatMultLayer(nn.Block):
    def __init__(self,units,in_units,**kwargs):
        super(GPUMatMultLayer,self).__init__(**kwargs)
        gpu_id = 0
        device = driver.Device(gpu_id)
        self.context = device.make_context(flags=driver.ctx_flags.SCHED_YIELD)        
        self.stream = driver.Stream()
        mod = driver.module_from_file(os.path.join(os.path.dirname(os.path.abspath(__file__)),'mat_mult.cubin'))
        
        self.kComputeMatMult = mod.get_function("kComputeMatMult")
        self.kComputeMatMult.prepare([np.int32,np.int32,np.int32,'P','P','P'])
        self.feature_dim = in_units
        self.output_dim = units
        with self.name_scope():
            self.weight = self.params.get('weight',shape=(in_units,units))
            self.bias = self.params.get('bias',shape=(units,))
        if self.context:
            self.context.pop()

    
    def forward(self,x):
        batch_size = x.shape[0]
        assert (batch_size % 96 == 0),'only support batch size with it remainder 96 is 0' 
        y = np.zeros((batch_size,self.output_dim))
        self.context.push()
        self.kComputeMatMult(np.int32(self.feature_dim), np.int32(batch_size), np.int32(self.output_dim),\
                driver.In(x.T.asnumpy().astype(np.float32)), driver.In(self.weight.data().asnumpy().astype(np.float32)),\
                driver.Out(y),block=(16,16,1), grid=(self.output_dim/TILE_SIZE ,batch_size/TILE_SIZE))
        self.context.pop()
        return nd.array(y)

class ClassifyLayer(nn.Block):
    def __init__(self,class_num=10,**kwargs):
        super(ClassifyLayer,self).__init__(**kwargs)
        self.classifyLayer = nn.Dense(class_num)

    def forward(self,x):
        return self.classifyLayer(x)

def net1():
    net = nn.Sequential()
    with net.name_scope():
        net.add(LeNetHeadLayer()) 
        net.add(nn.Dense(OUT_DIM))
        net.add(ClassifyLayer())
    return net


def net2():
    net = nn.Sequential()
    with net.name_scope():
        net.add(LeNetHeadLayer()) 
        net.add(GPUMatMultLayer(OUT_DIM,in_units=FEATURE_DIM))
        net.add(ClassifyLayer())
    return net

def get_net(netname):
    if netname == 'fc':
        return net1()
    elif netname == 'gpu_fc':
        return net2()
    else:
        raise ValueError('un supported net %s' % netname)
    

if __name__ == '__main__':
    print('fc with dense op %s' % get_net('fc'))
    print('fc with gpu matmult op %s' % get_net('gpu_fc'))


