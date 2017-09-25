#!/bin/python

from mxnet import autograd
from mxnet import gluon
from mxnet import nd
from mxnet import init
import mxnet as mx


from nets import get_net
import util

netname = 'gpu_fc'
epoches = 1
lr = 0.01

batch_size = 96
train_data, eval_data = util.load_minist_data(batch_size)

ctx = util.try_gpu()

net = get_net('gpu_fc')
print(net)
net.initialize(ctx=ctx)

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

trainer = gluon.Trainer(
    net.collect_params(), 'sgd', {'learning_rate':lr})

for epoch in range(epoches):
    train_loss = 0
    train_acc = 0
    idx = 0
    for data,label in train_data:
        #print('data shape %s, label shape %s' % (data.shape,label.shape))
        label = label.as_in_context(ctx)
        with autograd.record():
            output = net(data.as_in_context(ctx))
            loss = softmax_cross_entropy(output,label)
        loss.backward()
        trainer.step(batch_size)
        batch_loss = nd.mean(loss).asscalar()
        #print("Epoch %d batch %d loss %f" % (epoch, idx, batch_loss))
        idx +=1
        train_loss += batch_loss 
        train_acc += util.accuracy(output,label)

    eval_acc = util.evaluate_accuracy(eval_data, net, ctx)
    print("Epoch %d, Loss %f, Train acc %f, eval acc %f" % (epoch, train_loss/len(train_data), train_acc/len(train_data), eval_acc))

