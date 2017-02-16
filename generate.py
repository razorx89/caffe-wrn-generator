#!/usr/bin/env python
"""
The MIT License (MIT)

Copyright (c) 2016-2017 Sven Koitka (sven.koitka@fh-dortmund.de)

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

config = {}


def wide_basic(net, top, basename, num_input, num_output, stride):
    if config['bottleneck']:
        conv_params = [
            {'kernel_size': 1, 'stride': stride, 'pad': 0, 'num_output': num_output / 4},
            {'kernel_size': 3, 'stride': 1, 'pad': 1, 'num_output': num_output / 4},
            {'kernel_size': 1, 'stride': 1, 'pad': 0, 'num_output': num_output}
        ]
        dropout_layers = [1]
    else:
        conv_params = [
            {'kernel_size': 3, 'stride': stride, 'pad': 1, 'num_output': num_output},
            {'kernel_size': 3, 'stride': 1, 'pad': 1, 'num_output': num_output}
        ]
        dropout_layers = [1]

    resunit_layer = top
    shortcut_layer = top

    for i, p in enumerate(conv_params):
        branch_layer_name = '%sa_%d' % (basename, i + 1)
        add_dropout = i in dropout_layers and config['dropout']

        bn = L.BatchNorm(resunit_layer, in_place=i > 0)#, batch_norm_param={'use_global_stats': True})Bat
        scale = L.Scale(bn, in_place=True, scale_param={'bias_term': True})
        relu = L.ReLU(scale, in_place=True)
        if add_dropout:
            drop = L.Dropout(relu, in_place=True, dropout_ratio=config['dropout'])
        conv = L.Convolution(drop if add_dropout else relu, weight_filler={'type': 'msra'}, bias_term=False, **p)

        net[branch_layer_name + '_bn'] = bn
        net[branch_layer_name + '_scale'] = scale
        net[branch_layer_name + '_relu'] = relu
        if add_dropout:
            net[branch_layer_name + '_dropout'] = drop
        net[branch_layer_name + '_%dx%d_s%d' % (p['kernel_size'], p['kernel_size'], p['stride'])] = conv
        resunit_layer = conv

    if num_input != num_output:
        conv = L.Convolution(shortcut_layer, kernel_size=1, stride=stride, pad=0, num_output=num_output,
                             weight_filler={'type': 'xavier'}, bias_term=False)
        net['%sb_1x1_s%d' % (basename, stride)] = conv
        shortcut_layer = conv
    eltwise = L.Eltwise(resunit_layer, shortcut_layer, operation=P.Eltwise.SUM)
    net[basename] = eltwise
    return eltwise


def layer(net, top, basename, block, num_input, num_output, count, stride):
    top = block(net, top, basename + '_1', num_input, num_output, stride)
    for i in range(1, count):
        top = block(net, top, '%s_%d' % (basename, i+1), num_output, num_output, 1)
    return top


def wrn_cifar(num_residual_units, k, train_batch_size, test_batch_size, num_classes, db_train, db_test):
    assert k >= 1
    assert len(num_residual_units) == 3

    n = caffe.NetSpec()

    num_stages = [16, 16*k, 32*k, 64*k]

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

    top = layer(n, n.conv1, 'res2', wide_basic, num_stages[0], num_stages[1], num_residual_units[0], 1)
    top = layer(n, top, 'res3', wide_basic, num_stages[1], num_stages[2], num_residual_units[1], 2)
    top = layer(n, top, 'res4', wide_basic, num_stages[2], num_stages[3], num_residual_units[2], 2)

    n.bn_5 = L.BatchNorm(top)#, batch_norm_param={'use_global_stats': True})
    n.scale_5 = L.Scale(n.bn_5, in_place=True, scale_param={'bias_term': True})
    n.relu_5 = L.ReLU(n.scale_5, in_place=True)
    n.pool_5 = L.Pooling(n.relu_5, pool=P.Pooling.AVE, global_pooling=True)
    fc = L.InnerProduct(n.pool_5, name='fc%d' % num_classes, param=[dict(lr_mult=1, decay_mult=1),
                                                                    dict(lr_mult=2, decay_mult=0)],
                        num_output=num_classes, weight_filler={'type': 'gaussian', 'std': 0.01},
                        bias_filler={'type': 'constant', 'value': 0})
    n[fc.fn.params['name']] = fc

    n['score/loss'] = L.SoftmaxWithLoss(fc, n.label)
    n['score/top-1'] = L.Accuracy(fc, n.label, top_k=1)
    n['score/top-5'] = L.Accuracy(fc, n.label, top_k=5)

    depth = 1 + len(num_residual_units) + sum([x * 2 for x in num_residual_units])
    return str(n.to_proto()).replace('__test', ''), depth


def wrn_imagenet(num_residual_units, k, train_batch_size, test_batch_size, num_classes, db_train, db_test):
    assert k >= 1
    assert len(num_residual_units) == 4

    n = caffe.NetSpec()

    num_stages = [64, 256 * k, 512 * k, 1024 * k, 2048 * k]

    n.data, n.label = L.Data(batch_size=train_batch_size, include={'phase': caffe.TRAIN}, backend=P.Data.LMDB,
                             source=db_train, ntop=2,
                             transform_param={
                                 'mirror': True,
                                 'crop_size': 224,
                                 'mean_file': 'mean.binaryproto',
                                 'scale': 0.00390625
                             })
    n.data__test, n.label__test = L.Data(batch_size=test_batch_size, include={'phase': caffe.TEST},
                                         backend=P.Data.LMDB, source=db_test, ntop=2,
                                         transform_param={
                                             'mirror': False,
                                             'crop_size': 224,
                                             'mean_file': 'mean.binaryproto',
                                             'scale': 0.00390625
                                         })

    n.conv1 = L.Convolution(n.data, kernel_size=7, stride=2, pad=3, num_output=num_stages[0],
                            weight_filler={'type': 'msra'}, bias_term=False)
    n.pool1 = L.Pooling(n.conv1, pool=P.Pooling.MAX, stride=2, kernel_size=3)

    top = layer(n, n.pool1, 'res2', wide_basic, num_stages[0], num_stages[1], num_residual_units[0], 1)
    top = layer(n, top, 'res3', wide_basic, num_stages[1], num_stages[2], num_residual_units[1], 2)
    top = layer(n, top, 'res4', wide_basic, num_stages[2], num_stages[3], num_residual_units[2], 2)
    top = layer(n, top, 'res5', wide_basic, num_stages[3], num_stages[4], num_residual_units[3], 2)

    n.bn_6 = L.BatchNorm(top)#, batch_norm_param={'use_global_stats': True})
    n.scale_6 = L.Scale(n.bn_6, in_place=True, scale_param={'bias_term': True})
    n.relu_6 = L.ReLU(n.scale_6, in_place=True)
    n.pool_6 = L.Pooling(n.relu_6, pool=P.Pooling.AVE, global_pooling=True)
    fc = L.InnerProduct(n.pool_6, name='fc%d' % num_classes, param=[dict(lr_mult=1, decay_mult=1),
                                                                    dict(lr_mult=2, decay_mult=0)],
                        num_output=num_classes, weight_filler={'type': 'gaussian', 'std': 0.01},
                        bias_filler={'type': 'constant', 'value': 0})
    n[fc.fn.params['name']] = fc

    n['score/loss'] = L.SoftmaxWithLoss(fc, n.label)
    n['score/top-1'] = L.Accuracy(fc, n.label, top_k=1)
    n['score/top-5'] = L.Accuracy(fc, n.label, top_k=5)

    depth = 1 + len(num_residual_units) + sum([x * 2 for x in num_residual_units])
    return str(n.to_proto()).replace('__test', ''), depth

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=['cifar10', 'cifar100', 'imagenet'])
    parser.add_argument('num_residual_units', help='Number of residual units per spatial resolution (e.g. "3,3,3" for CIFAR-10/CIFAR-100)')
    parser.add_argument('k', type=int, help='Widening factor for convolutional layers')
    parser.add_argument('--dropout', type=float, help='If specifed then a dropout layer is inserted between convolutional layers in a residual unit')
    parser.add_argument('--batch-size-train', type=int, help='Batch size for the training phase')
    parser.add_argument('--batch-size-test', type=int, help='Batch size for the test phase')
    parser.add_argument('--lmdb-train', help='Path to the training LMDB')
    parser.add_argument('--lmdb-test', help='Path to the test LMDB')
    parser.add_argument('--bottleneck-resunit', default=False, action='store_true',
                        help='Uses 1x1, 3x3, 1x1 convolutional layers with reduction of feature maps')
    args = parser.parse_args()

    config['bottleneck'] = args.bottleneck_resunit
    config['dropout'] = args.dropout

    assert args.k >= 1

    num_residual_units = [int(x) for x in args.num_residual_units.split(',')]

    if args.dataset in ['cifar10', 'cifar100']:
        np, depth = wrn_cifar(num_residual_units=num_residual_units, k=args.k, train_batch_size=args.batch_size_train or 128,
                       test_batch_size=args.batch_size_test or 100, num_classes=10 if args.dataset == 'cifar10' else 100,
                       db_train=args.lmdb_train or 'db_train_data.lmdb', db_test=args.lmdb_test or 'db_test_data.lmdb')
    elif args.dataset == 'imagenet':
        np, depth = wrn_imagenet(num_residual_units=num_residual_units, k=args.k, train_batch_size=args.batch_size_train or 32,
                       test_batch_size=args.batch_size_test or 10, num_classes=1000,
                       db_train=args.lmdb_train or 'db_train_data.lmdb', db_test=args.lmdb_test or 'db_test_data.lmdb')

    prototxt_path = '%s_WRN-%d-%d' % (args.dataset, depth, args.k)
    if config['bottleneck']:
        prototxt_path += '_bottleneck'
    if config['dropout']:
        prototxt_path += '_dropout'
    prototxt_path += '_train_val.prototxt'

    with open(prototxt_path, 'w') as ofile:
        ofile.write(np)
