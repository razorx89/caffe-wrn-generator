#!/usr/bin/env python
"""
The MIT License (MIT)

Copyright (c) 2016 Sven Koitka (sven.koitka@fh-dortmund.de)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import argparse
import caffe
from caffe import layers as L
from caffe import params as P


def wide_basic(net, basename, num_input, num_output, stride, dropout):
    conv_params = [
        {'kernel_size': 3, 'stride': stride, 'pad': 1},
        {'kernel_size': 3, 'stride': 1, 'pad': 1}
    ]

    prev_layer = net.tops[net.tops.keys()[-1]]
    prev_layer2 = prev_layer

    for i, p in enumerate(conv_params):
        if i == 0:
            bn = L.BatchNorm(prev_layer)
            relu = L.ReLU(bn, in_place=True)
            conv = L.Convolution(relu, num_output=num_output, weight_filler={'type': 'msra'}, bias_term=False, **p)

            net[basename + 'a_1_bn'] = bn
            net[basename + 'a_1_relu'] = relu
            net[basename + 'a_1_%dx%d_s%d' % (p['kernel_size'], p['kernel_size'], p['stride'])] = conv
            prev_layer = conv
        else:
            bn = L.BatchNorm(prev_layer, in_place=True)
            relu = L.ReLU(bn, in_place=True)
            if dropout:
                drop = L.Dropout(relu, in_place=True, dropout_ratio=dropout)
            conv = L.Convolution(drop if dropout else relu, num_output=num_output, weight_filler={'type': 'msra'},
                                 bias_term=False, **p)

            net[basename + 'a_2_bn'] = bn
            net[basename + 'a_2_relu'] = relu
            if dropout:
                net[basename + 'a_2_dropout'] = drop
            net[basename + 'a_2_%dx%d_s%d' % (p['kernel_size'], p['kernel_size'], p['stride'])] = conv
            prev_layer = conv

    if num_input != num_output:
        conv = L.Convolution(prev_layer2, kernel_size=1, stride=stride, pad=0, num_output=num_output,
                             weight_filler={'type': 'xavier'}, bias_term=False)
        net['%sb_1x1_s%d' % (basename, stride)] = conv
        prev_layer2 = conv
    eltwise = L.Eltwise(prev_layer, prev_layer2, operation=P.Eltwise.SUM)
    net[basename] = eltwise


def layer(net, basename, block, num_input, num_output, count, stride, dropout):
    block(net, basename + '_1', num_input, num_output, stride, dropout)
    for i in range(1, count):
        block(net, '%s_%d' % (basename, i+1), num_output, num_output, 1, dropout)


def wrn_cifar(depth, k, train_batch_size, test_batch_size, num_classes, db_train, db_test, dropout):
    assert k >= 1
    assert (depth - 4) % 6 == 0

    n = caffe.NetSpec()

    num_stages = [16, 16*k, 32*k, 64*k]
    num_residual_units = (depth - 4) / 6

    n.data, n.label = L.Data(batch_size=train_batch_size, include={'phase': caffe.TRAIN}, backend=P.Data.LMDB,
                             source=db_train, ntop=2,
                             transform_param={
                                 'mirror': True,
                                 'crop_size': 32,
                                 'mean_file': 'mean.binaryproto',
                                 'scale': 0.00390625
                             })
    n.data__test, n.label__test = L.Data(batch_size=test_batch_size, include={'phase': caffe.TEST},
                                         backend=P.Data.LMDB, source=db_test, ntop=2,
                                         transform_param={
                                             'mirror': False,
                                             'crop_size': 32,
                                             'mean_file': 'mean.binaryproto',
                                             'scale': 0.00390625
                                         })

    n.conv1 = L.Convolution(n.data, kernel_size=3, stride=1, pad=1, num_output=num_stages[0],
                            weight_filler={'type': 'msra'}, bias_term=False)

    layer(n, 'res2', wide_basic, num_stages[0], num_stages[1], num_residual_units, 1, dropout)
    layer(n, 'res3', wide_basic, num_stages[1], num_stages[2], num_residual_units, 2, dropout)
    layer(n, 'res4', wide_basic, num_stages[2], num_stages[3], num_residual_units, 2, dropout)

    prev_layer = n.tops[n.tops.keys()[-1]]
    n.bn_5 = L.BatchNorm(prev_layer)
    n.relu_5 = L.ReLU(n.bn_5, in_place=True)
    n.pool_5 = L.Pooling(n.relu_5, pool=P.Pooling.AVE, global_pooling=True)
    fc = L.InnerProduct(n.pool_5, name='fc%d' % num_classes, param=[dict(lr_mult=1, decay_mult=1),
                                                                    dict(lr_mult=2, decay_mult=0)],
                        num_output=num_classes, weight_filler={'type': 'gaussian', 'std': 0.01},
                        bias_filler={'type': 'constant', 'value': 0})
    n[fc.fn.params['name']] = fc

    n['score/loss'] = L.SoftmaxWithLoss(fc, n.label)
    n['score/top-1'] = L.Accuracy(fc, n.label, top_k=1)
    n['score/top-5'] = L.Accuracy(fc, n.label, top_k=5)

    return str(n.to_proto()).replace('__test', '')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=['cifar10', 'cifar100'])
    parser.add_argument('depth', type=int, help='Depth of the entire network (for CIFAR-10/CIFAR-100: (depth-4)%%6 == 0)')
    parser.add_argument('k', type=int, help='Widening factor for convolutional layers')
    parser.add_argument('--dropout', type=float, help='If specifed then a dropout layer is inserted between convolutional layers in a residual unit')
    parser.add_argument('--batch-size-train', type=int, help='Batch size for the training phase')
    parser.add_argument('--batch-size-test', type=int, help='Batch size for the test phase')
    parser.add_argument('--lmdb-train', help='Path to the training LMDB')
    parser.add_argument('--lmdb-test', help='Path to the test LMDB')
    args = parser.parse_args()

    assert args.k >= 1

    prototxt_path = '%s_WRN-%d-%d' % (args.dataset, args.depth, args.k)
    if args.dropout:
        prototxt_path += '_dropout'
    prototxt_path +=  '_train_val.prototxt'

    if args.dataset in ['cifar10', 'cifar100']:
        np = wrn_cifar(depth=args.depth, k=args.k, train_batch_size=args.batch_size_train or 128,
                       test_batch_size=args.batch_size_test or 100, num_classes=10 if args.dataset == 'cifar10' else 100,
                       db_train=args.lmdb_train or 'db_train_data.lmdb', db_test=args.lmdb_test or 'db_test_data.lmdb',
                       dropout=args.dropout)

    with open(prototxt_path, 'w') as ofile:
        ofile.write(np)
