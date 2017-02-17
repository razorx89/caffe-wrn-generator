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


class WideResidualNetworkGenerator:
    def __init__(self, config):
        self.__config = config
        self.__num_residual_units = config['num-residual-units']

        # Check if a sufficient number of residual units was passed to the generator for the chosen dataset
        if config['dataset'] == 'imagenet':
            assert len(self.__num_residual_units) == 4
        else:
            assert len(self.__num_residual_units) == 3

        # When using bottneck residual units the overall feature map count increases, whereas the feature map count
        # within a residual unit block is reduced by the same factor
        k = config['widen-factor']
        if config['dataset'] == 'imagenet':
            if config['bottleneck']:
                self.__num_feature_maps = [64, 256 * k, 512 * k, 1024 * k, 2048 * k]
            else:
                self.__num_feature_maps = [64, 64 * k, 128 * k, 256 * k, 512 * k]
        else:
            if config['bottleneck']:
                self.__num_feature_maps = [16, 64 * k, 128 * k, 256 * k]
            else:
                self.__num_feature_maps = [16, 16 * k, 32 * k, 64 * k]

    @property
    def depth(self):
        num_conv_per_resunit = 3 if config['bottleneck'] else 2
        num_conv_for_resunits = sum([x * num_conv_per_resunit for x in self.__num_residual_units])
        num_shortcut_conv = len(self.__num_residual_units)
        return 1 + num_shortcut_conv + num_conv_for_resunits  # + 1 for first convolutional layer

    def __wide_basic(self, net, top, basename, num_input, num_output, stride, generate_deploy):
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

            if generate_deploy:
                bn = L.BatchNorm(resunit_layer, in_place=i > 0, batch_norm_param={'use_global_stats': True})
            else:
                bn = L.BatchNorm(resunit_layer, in_place=i > 0)
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

    def __layer(self, net, top, basename, block, num_input, num_output, count, stride, generate_deploy):
        top = block(net, top, basename + '_1', num_input, num_output, stride, generate_deploy)
        for i in range(1, count):
            top = block(net, top, '%s_%d' % (basename, i+1), num_output, num_output, 1, generate_deploy)
        return top

    def __create_data_layer(self, n, deploy):
        if deploy:
            n.data = L.Input(input_param={
                'shape': [
                    {
                        'dim': [
                            self.__config['deploy-batch-size'],
                            self.__config['deploy-num-channels'],
                            self.__config['crop-size'],
                            self.__config['crop-size']
                        ]
                    }
                ]
            })
        else:
            transform_param = {
                'mirror': True,
                'crop_size': self.__config['crop-size'],
                'scale': 0.00390625
            }
            if self.__config['mean-file']:
                transform_param['mean_file'] = self.__config['mean-file']

            n.data, n.label = L.Data(batch_size=self.__config['train-batch-size'],
                                     include={'phase': caffe.TRAIN},
                                     backend=P.Data.LMDB,
                                     source=self.__config['train-database'],
                                     ntop=2,
                                     transform_param=transform_param)

            transform_param = dict.copy(transform_param)
            transform_param['mirror'] = False
            n.data__test, n.label__test = L.Data(batch_size=self.__config['test-batch-size'],
                                                 include={'phase': caffe.TEST},
                                                 backend=P.Data.LMDB,
                                                 source=self.__config['test-database'],
                                                 ntop=2,
                                                 transform_param=transform_param)
        return n.data

    def __create_first_convolutional_layer(self, n, bottom):
        if self.__config['dataset'] == 'imagenet':
            n.conv1 = L.Convolution(bottom, kernel_size=7, stride=2, pad=3, num_output=self.__num_feature_maps[0],
                                    weight_filler={'type': 'msra'}, bias_term=False)
            n.pool1 = L.Pooling(n.conv1, pool=P.Pooling.MAX, stride=2, kernel_size=3)
            return n.pool1
        else:
            n.conv1 = L.Convolution(bottom, kernel_size=3, stride=1, pad=1, num_output=self.__num_feature_maps[0],
                                    weight_filler={'type': 'msra'}, bias_term=False)
            return n.conv1

    def __create_output_layer(self, n, bottom, deploy):
        if deploy:
            n['prob'] = L.Softmax(bottom)
        else:
            n['score/loss'] = L.SoftmaxWithLoss(bottom, n.label)
            n['score/top-1'] = L.Accuracy(bottom, n.label, top_k=1)
            n['score/top-5'] = L.Accuracy(bottom, n.label, top_k=5)

    def __create_fully_connected_layer(self, n, bottom, deploy):
        if self.__config['dataset'] == 'imagenet':
            layer_idx = 6
        else:
            layer_idx = 5

        if deploy:
            bn = L.BatchNorm(bottom, batch_norm_param={'use_global_stats': True})
        else:
            bn = L.BatchNorm(bottom)
        scale = L.Scale(bn, in_place=True, scale_param={'bias_term': True})
        relu = L.ReLU(scale, in_place=True)
        pool = L.Pooling(relu, pool=P.Pooling.AVE, global_pooling=True)
        fc = L.InnerProduct(pool,
                            param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                            num_output=self.__config['num-classes'], weight_filler={'type': 'gaussian', 'std': 0.01},
                            bias_filler={'type': 'constant', 'value': 0})

        n['bn_%d' % layer_idx] = bn
        n['scale_%d' % layer_idx] = scale
        n['relu_%d' % layer_idx] = relu
        n['pool_%d' % layer_idx] = pool
        n['fc%d' % self.__config['num-classes']] = fc
        return fc

    def __create_network_prototxt(self, deploy):
        n = caffe.NetSpec()
        top = self.__create_data_layer(n, deploy=deploy)
        top = self.__create_first_convolutional_layer(n, top)

        # Create Residual Units
        for i in range(len(self.__num_residual_units)):
            stride = 1 if i == 0 else 2
            top = self.__layer(n, top, 'res%d' % (2+i), self.__wide_basic, self.__num_feature_maps[i],
                               self.__num_feature_maps[i+1], self.__num_residual_units[i], stride, deploy)

        top = self.__create_fully_connected_layer(n, top, deploy=deploy)
        self.__create_output_layer(n, top, deploy=deploy)

        # Return prototxt
        return str(n.to_proto()).replace('__test', '')

    def get_train_val(self):
        return self.__create_network_prototxt(deploy=False)

    def get_deploy(self):
        return self.__create_network_prototxt(deploy=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=['cifar10', 'cifar100', 'imagenet'])
    parser.add_argument('num_residual_units', help='Number of residual units per spatial resolution (e.g. "3,3,3" for CIFAR-10/CIFAR-100)')
    parser.add_argument('k', type=int, help='Widening factor for convolutional layers')
    parser.add_argument('--dropout', type=float, help='If specifed then a dropout layer is inserted between convolutional layers in a residual unit')
    parser.add_argument('--train-batch-size', type=int, help='Batch size for the training phase')
    parser.add_argument('--test-batch-size', type=int, help='Batch size for the test phase')
    parser.add_argument('--deploy-batch-size', type=int, help='Batch size for the deploy network')
    parser.add_argument('--deploy-num-channels', type=int, default=3, help='Number of image channels for deployment input layer')
    parser.add_argument('--train-database', help='Path to the training LMDB')
    parser.add_argument('--test-database', help='Path to the test LMDB')
    parser.add_argument('--bottleneck-resunit', default=False, action='store_true',
                        help='Uses 1x1, 3x3, 1x1 convolutional layers with reduction of feature maps')
    parser.add_argument('--no-deploy-prototxt', default=False, action='store_true')
    parser.add_argument('--no-train-val-prototxt', default=False, action='store_true')
    parser.add_argument('--crop-size', type=int, help='Size of the random crop image during training')
    parser.add_argument('--mean-file')
    args = parser.parse_args()

    assert args.k >= 1

    # Build configuration for network generation
    config = {
        'bottleneck': args.bottleneck_resunit,
        'dataset': args.dataset,
        'dropout': args.dropout,
        'train-database': args.train_database or 'db_train_data.lmdb',
        'test-database': args.test_database or 'db_test_data.lmdb',
        'deploy-num-channels': args.deploy_num_channels,
        'mean-file': args.mean_file,
        'widen-factor': args.k,
        'num-residual-units': [int(x) for x in args.num_residual_units.split(',')]
    }

    # Define dataset specific defaults if not otherwise given
    if args.dataset in ['cifar10', 'cifar100']:
        config['train-batch-size'] = args.train_batch_size or 128
        config['test-batch-size'] = args.test_batch_size or 100
        config['deploy-batch-size'] = args.deploy_batch_size or 128
        config['num-classes'] = 10 if args.dataset == 'cifar10' else 100
        config['crop-size'] = args.crop_size or 32
    elif args.dataset == 'imagenet':
        config['train-batch-size'] = args.train_batch_size or 32
        config['test-batch-size'] = args.test_batch_size or 10
        config['deploy-batch-size'] = args.deploy_batch_size or 32
        config['num-classes'] = 1000
        config['crop-size'] = args.crop_size or 224

    # Create generator with configuration
    generator = WideResidualNetworkGenerator(config)
    depth = generator.depth

    # Determine prototxt prefix
    prototxt_path = '%s_WRN-%d-%d' % (args.dataset, depth, args.k)
    if config['bottleneck']:
        prototxt_path += '_bottleneck'
    if config['dropout']:
        prototxt_path += '_dropout'

    # Generate both train-val and deploy prototxt specifications
    if not args.no_train_val_prototxt:
        net_train_val = generator.get_train_val()
        with open(prototxt_path + '_train_val.prototxt', 'w') as ofile:
            ofile.write(net_train_val)

    if not args.no_deploy_prototxt:
        net_deploy = generator.get_deploy()
        with open(prototxt_path + '_deploy.prototxt', 'w') as ofile:
            ofile.write(net_deploy)
