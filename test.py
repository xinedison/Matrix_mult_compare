#!/bin/python

import os
import numpy as np

from util import Timer

import mxnet as mx
from mxnet import nd

gpu_id = 0
os.environ['CUDA_VISIBLE_DEVICES'] = '' # str(gpu_id)
import tensorflow as tf

def init_cubin():
    from pycuda import driver
    from pycuda import gpuarray
    from pycuda import autoinit

    global context,kComputeMatMult
    device=driver.Device(gpu_id)
    context=device.make_context(flags=driver.ctx_flags.SCHED_YIELD)        
    stream=driver.Stream()
    #print model_path
    mod=driver.module_from_file(os.path.join(os.path.dirname(os.path.abspath(__file__)),'mat_mult.cubin'))
    
    kComputeMatMult = mod.get_function("kComputeMatMult")
    kComputeMatMult.prepare([np.int32,np.int32,np.int32,'P','P','P'])
    
    if context:
        context.pop()
    print 'init ok'

TILE_SIZE = 96
PAD_QUERY_NUM = 96

#init_cubin()

def print_shape(data,des=''):
    print("%s shape is %s" % (des,data.shape))

def compute(batch, weight, fc_act):
    '''
    Args :
        batch : input data of shape feature_dim x batch_size 
        weight : fc layer weight , feature_dim x fc output layer dim
        fc_act : fc_act result , batch_size x fc output layer dim
    '''
    #gpu_weight = gpuarray.to_gpu(weight)
    #query_batch_gpu = gpuarray.to_gpu_async(batch,stream=stream)
    #query_batch_gpu = gpuarray.to_gpu(batch)

    #print("batch shape %s with data in  compute %s" % (batch.shape,batch))
    #print("weight shape %s data in compuate %s" % (weight.shape,weight))

    context.push()
    feature_dim,num_output = weight.shape
    batch_feature_dim,batch_size = batch.shape
    assert batch_feature_dim == feature_dim,'feature dim should be equal!!!'
    #kComputeMatMult.prepared_async_call(\
    #            (num_output/TILE_SIZE ,PAD_QUERY_NUM/TILE_SIZE), (16,16,1), \
    #            stream, \
    #            np.int32(feature_dim), np.int32(PAD_QUERY_NUM), np.int32(num_output), \
    #            query_batch_gpu.gpudata, gpu_weight.gpudata, fc_act.gpudata)

    with Timer("Mat Cubin compute"):
        kComputeMatMult(np.int32(feature_dim), np.int32(batch_size), np.int32(num_output), driver.In(batch), driver.In(weight),\
             driver.Out(fc_act),block=(16,16,1), grid=(num_output/TILE_SIZE ,batch_size/TILE_SIZE))
    context.pop()
    return fc_act
    

def is_equal(a,b,method='any',epsilon=0.001):
    '''
    Judge whether a and b matrix is equal
    '''
    if not (a.shape == b.shape):
        return False

    if method == 'all':
        return (a==b).all()
    elif method == 'any':
        return not (((a-b)>epsilon).any())
    else:
        raise ValueError("equal compare type %s is invalid,should by all or any" % method)

def mx_matrix_mult(batch,weight,ctx=None):
    '''
    Args:
        batch : feature_dim x batch_size search data
        weight : feature_dim x output_dim weight data
    Return:
        matrix multiply result batch_size x out_dim
    '''

    nd_batch = nd.array(batch.T, ctx)
    nd_weight = nd.array(weight, ctx)
    with Timer('mxnet dot compute'):
        nd_result = nd.dot(nd_batch, nd_weight)
    return nd_result.asnumpy().astype(np.float32)

def tf_matrix_mult(batch,weight,ctx=''):
        
    feature_dim,batch_size = batch.shape
    w_feature_dim, out_dim = weight.shape
    assert feature_dim == w_feature_dim, 'feature dim not equal'
    x = tf.placeholder(tf.float32,shape=(batch_size, feature_dim))
    y = tf.placeholder(tf.float32,shape=(feature_dim, out_dim))
    x_malt_y = tf.matmul(x,y)
    with tf.Session() as sess:
        with Timer('TF %s compute' % ctx):
            result = sess.run(x_malt_y, feed_dict={x:batch.T, y:weight})
    return result

def test_1(batch_size,num_output,feature_dim):
    batch = np.arange(feature_dim*batch_size).reshape((feature_dim, batch_size)).astype(np.float32)
    weight = np.arange(feature_dim*num_output).reshape((feature_dim,num_output)).astype(np.float32)
    distance = np.zeros((batch_size, num_output), np.float32)
    with Timer('Test 1 Gpu compute'):
        distance = compute(batch,weight,distance)
    #print('result from gpu %s' % res)
    
    with Timer('Test 1 numpy dot'):
        real_dis = np.dot(batch.T,weight)
    #real_dis.astype(np.int64)
    #print('real dis shape %s data is %s' % (real_dis.shape,real_dis))
    assert (distance==real_dis).all(),"test_1 batch_size %d, num_ouput %d, feature_dim %d failed!!!" % (batch_size,num_output,feature_dim)
    



def test_2(batch_size,num_output,feature_dim):
    print('Current test for batch size %d, num_output %d, feature_dim %d' % (batch_size, num_output, feature_dim)) 
    batch = np.random.random((feature_dim, batch_size)).astype(np.float32)
    weight = np.random.random((feature_dim, num_output)).astype(np.float32)
    mult_res = np.zeros((batch_size,num_output), np.float32)

    with Timer('---test_2 numpy mult'):
        real_res = np.dot(batch.T,weight)

    #with Timer('---test_2 Gpu matmult'):
    #    mult_res = compute(batch,weight, mult_res)
    
    #with Timer('---test_2 mxnet nd GPU dot'):
    #    nd_gpu_result = mx_matrix_mult(batch, weight, mx.gpu(gpu_id))

    with Timer('---test_2 nd CPU mult'):
        nd_cpu_result = mx_matrix_mult(batch, weight)
    
    with Timer('---test_2 TF matmul'):
        tf_result = tf_matrix_mult(batch, weight,ctx='cpu')

    #assert is_equal(mult_res, real_res), "Gpu dist is not equal numpy dis!!!" 
    #assert is_equal(nd_gpu_result, real_res), "Mxnet GPU dist not equal to numpy dis!!!"
    assert is_equal(nd_cpu_result, real_res), "Mxnet CPU dist not equal to numpy dis!!!"
    assert is_equal(tf_result, real_res), "TF malt result is not correct!!!"

    #assert is_equal(nd_gpu_result, mult_res), "Mxnet nd gpu dist is not equal to gpu mat mult dis!!!"
    print('test ok')


def test_cases():
    for batch_mult in [1,2,3]:
        for feature_dim in [128,256,512]:
            for output_dim in [96000, 192000, 288000, 384000]:
                test_2(PAD_QUERY_NUM*batch_mult, output_dim, feature_dim)

#test_1(PAD_QUERY_NUM,96,512)
#test_2(PAD_QUERY_NUM*3, 96000, 512)
test_cases()

print('run ok\n')

