#!/bin/python

from pycuda import driver
from pycuda import gpuarray
from pycuda import autoinit

import os
import numpy as np
from util import Timer

gpu_id = 2
device=driver.Device(gpu_id)
context=device.make_context(flags=driver.ctx_flags.SCHED_YIELD)        
stream=driver.Stream()
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'mat_mult.cubin')
#print model_path
mod=driver.module_from_file(model_path)

kComputeMatMult = mod.get_function("kComputeMatMult")
kComputeMatMult.prepare([np.int32,np.int32,np.int32,'P','P','P'])

print 'init ok'

TILE_SIZE = 96
PAD_QUERY_NUM = 96


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
    #kComputeMatMult.prepared_async_call(\
    #            (num_output/TILE_SIZE ,PAD_QUERY_NUM/TILE_SIZE), (16,16,1), \
    #            stream, \
    #            np.int32(feature_dim), np.int32(PAD_QUERY_NUM), np.int32(num_output), \
    #            query_batch_gpu.gpudata, gpu_weight.gpudata, fc_act.gpudata)

    kComputeMatMult(np.int32(feature_dim), np.int32(PAD_QUERY_NUM), np.int32(num_output), driver.In(batch), driver.In(weight),\
             driver.Out(fc_act),block=(16,16,1), grid=(num_output/TILE_SIZE ,PAD_QUERY_NUM/TILE_SIZE))
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
    distance = np.zeros((batch_size,num_output), np.float32)

    with Timer('Test 2 Gpu matmult'):
        distance = compute(batch,weight, distance)
    

    with Timer('Test 2 numpy mult'):
        real_dis = np.dot(batch.T,weight)
    
    #print('numpy dis %s' % real_dis)

    
    from mxnet import nd
    import mxnet as mx
    nd_batch_gpu_t = nd.array(batch.T, mx.gpu(gpu_id))
    nd_weight_gpu = nd.array(weight, mx.gpu(gpu_id))
    with Timer('Test 2 mxnet GPU dot compute'):
        nd_dis_gpu = nd.dot(nd_batch_gpu_t, nd_weight_gpu)
    #print('mxnet dis %s' % nd_dis_gpu)

    nd_batch_cpu_t = nd.array(batch.T, mx.cpu())
    nd_weight_cpu = nd.array(weight, mx.cpu())
    with Timer('Test 2 mxnet cpu dot'):
        nd_dis_cpu = nd.dot(nd_batch_cpu_t, nd_weight_cpu)
    
    assert is_equal(distance, real_dis), "Gpu dist is not equal numpy dis!!!" 
    assert is_equal(nd_dis_gpu.asnumpy().astype(np.float32), real_dis), "Mxnet GPU dist not equal to numpy dis!!!"
    assert is_equal(nd_dis_gpu.asnumpy().astype(np.float32), distance), "Mxnet nd gpu dist is not equal to gpu mat mult dis!!!"
    assert is_equal(nd_dis_cpu.asnumpy().astype(np.float32), real_dis), "Mxnet CPU dist not equal to numpy dis!!!"
    print('test ok')


def test_cases():
    for feature_dim in [128,256,512]:
        for output_dim in [96000, 192000, 288000, 384000]:
            test_2(PAD_QUERY_NUM, output_dim, feature_dim)

#test_1(PAD_QUERY_NUM,96,512)
test_2(PAD_QUERY_NUM, 96000, 512)
test_cases()

if context:
    context.pop()

print('run ok\n')

