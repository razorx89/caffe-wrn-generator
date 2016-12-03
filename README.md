Caffe Wide-Residual-Network (WRN) Generator
===========================================

This generator is a reimplementation of Wide Residual Networks (WRN).
Full-Pre-Activation Residual Units from [[2]](#ref2) are used with two
convolutional units of size 3x3 per residual unit. Support for bottleneck
residual units (3 convolutional layers: 1x1, 3x3, 1x1) will be added in the
future. Currently the generator is implemented only for CIFAR-10/CIFAR-100, but
ImageNet will be added soon.

How to use
----------
__WRN-16-4 for CIFAR-10:__  
Command: `python generate.py cifar10 16 4`  
Output: cifar10_WRN-16-4_train_val.prototxt

__WRN-16-4 with Dropout for CIFAR-100:__  
Command: `python generate.py cifar100 16 4 --dropout=0.3`  
Output: cifar100_WRN-16-4_dropout_train_val.prototxt

For more custumization options check the possible arguments with
`python generate.py --help`.

References
----------
- <a name='ref1'></a>[1] Sergey Zagoruyko, Nikos Komodakis; "Wide Residual
  Networks"; British Machine Vision Conference (BMVC) 2016, 19-22 September,
  York, UK; 2016; [arXiv](https://arxiv.org/abs/1605.07146),
  [Github](https://github.com/szagoruyko/wide-residual-networks)
- <a name='ref2'></a>[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun;
  "Identity Mappings in Deep Residual Networks", arXiv preprint arXiv:1603.05027,
  2016; [arXiv](https://arxiv.org/abs/1603.05027),
  [Github](https://github.com/KaimingHe/resnet-1k-layers)

Visualization of a WRN-16-4 with Dropout
-----------------------------
![CIFAR-100 WRN-16-4 /w Dropout visualization](example/cifar100_WRN-16-4_dropout_net.png?raw=true)
