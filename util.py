# -*- coding: utf-8 -*-
'''
Created on 2015年5月8日

'''
import logging
logger = logging.getLogger('ugsearch')
formatter = logging.Formatter('%(asctime)s-%(levelname)s:%(message)s')

def __initLogger__(logger, formatter):
    logger.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)

__initLogger__(logger, formatter)

import time
class Timer(object):
    def __init__(self, fname):
        self.fname = fname

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs
        #global logger
        #logger.info('%s elapsed time: %f ms' % (self.fname, self.msecs))
        print('%s elapsed time: %f ms' % (self.fname, self.msecs))

import mxnet as mx
from mxnet import ndarray as nd
from mxnet import gluon
from mxnet import image

def SGD(params,lr):
    for param in params:
        param[:] = param - lr*param.grad


def attach_grad_for_params(params):
    '''
    add gradient and make params to be compute graph nodes
    '''
    for param in params:
        param.attach_grad()

def accuracy(output,label):
    return nd.mean(nd.argmax(output,axis=1) == label).asscalar()


def evaluate_accuracy(data_iter, net, ctx=mx.cpu()):
    acc = 0
    for data,label in data_iter:
        output = net(data.as_in_context(ctx))
        acc += accuracy(output,label.as_in_context(ctx))
    return acc/len(data_iter)

def load_minist_data(batch_size):
    def transform(data,label):
        data = image.imresize(data,224,224)
        return nd.transpose(data.astype('float32'),(2,0,1))/255,label.astype('float32')

    mnist_train = gluon.data.vision.MNIST(train=True, transform=transform)
    mnist_test = gluon.data.vision.MNIST(train=False, transform=transform)
    train_data = gluon.data.DataLoader(mnist_train, batch_size, shuffle=True)
    eval_data = gluon.data.DataLoader(mnist_test, batch_size, shuffle=False)
    return train_data, eval_data


def try_gpu():
    """
    If GPU is available, return mx.gpu(0); 
    else return mx.cpu()
    """
    try:
        ctx = mx.gpu()
        _ = nd.zeros((1,), ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx
