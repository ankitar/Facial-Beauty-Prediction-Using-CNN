import numpy as np
import matplotlib.pyplot as plt
from pylab import *

# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples

import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

net = caffe.Classifier('examples/face_feature_extraction/facenet_deploy.prototxt',
                       'examples/face_feature_extraction/facenet_iter_500.caffemodel')
net.set_phase_test()
net.set_mode_gpu()
# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
#net.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'))  # ImageNet mean
#net.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
net.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

scores = net.predict([caffe.io.load_image('data/face_feature_extraction/mixture/506.jpg')])

[(k, v.data.shape) for k, v in net.blobs.items()]

for k, v in net.params.items():
 print (k, v[0].data.shape) 


# take an array of shape (n, height, width) or (n, height, width, channels)
#  and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data)


# index four is the center crop
plt.imshow(net.deprocess('data', net.blobs['data'].data[0]))
savefig('506.png')

filters = net.params['conv1'][0].data
vis_square(filters.transpose(0, 2, 3, 1))
savefig('506afterConv1.png')

feat = net.blobs['conv1'].data[0, :48]
vis_square(feat, padval=1)
savefig('506afterRectifiedConv1.png')


"""
filters = net.params['pool1'][0].data
vis_square(filters[:48].reshape(48**2, 5, 5))
savefig('506afterPool1.png')

feat = net.blobs['pool1'].data[0, :36]
vis_square(feat, padval=1)
savefig('506afterRectifiedPool1.png')

feat = net.blobs['conv2'].data[0]
vis_square(feat, padval=0.5)
savefig('506afterConv2.png')

feat = net.blobs['pool2'].data[0]
vis_square(feat, padval=0.5)
savefig('506afterPool2.png')

feat = net.blobs['conv3'].data[0]
vis_square(feat, padval=0.5)
savefig('506afterConv3.png')


feat = net.blobs['pool3'].data[0]
vis_square(feat, padval=0.5)
savefig('506afterPool3.png')

feat = net.blobs['conv4'].data[0]
vis_square(feat, padval=0.5)
savefig('506afterConv4.png')

feat = net.blobs['pool4'].data[0]
vis_square(feat, padval=0.5)
savefig('506afterPool4.png')
"""
feat = net.blobs['pool2'].data[0]
vis_square(feat, padval=0.5)
savefig('506afterPool2.png')

feat = net.blobs['conv2'].data[0,:16]
vis_square(feat, padval=0.5)
savefig('506afterConv2.png')

feat = net.blobs['conv3'].data[0,:16]
vis_square(feat, padval=0.5)
savefig('506afterConv3.png')


feat = net.blobs['pool3'].data[0]
vis_square(feat, padval=1)
savefig('506afterPool3.png')

feat = net.blobs['conv4'].data[0, :14]
plt.subplot(2, 1, 1)
plt.plot(feat.flat)
plt.subplot(2, 1, 2)
_ = plt.hist(feat.flat[feat.flat > 0], bins=100)
savefig('histogramOfPositiveValuesafterConv4.png')

feat = net.blobs['pool4'].data[0]
plt.subplot(2, 1, 1)
plt.plot(feat.flat)
plt.subplot(2, 1, 2)
_ = plt.hist(feat.flat[feat.flat > 0], bins=100)
savefig('histogramOfPositiveValuesafterPool4.png')

feat = net.blobs['conv5'].data[0, :10]
plt.subplot(2, 1, 1)
plt.plot(feat.flat)
plt.subplot(2, 1, 2)
_ = plt.hist(feat.flat[feat.flat > 0], bins=100)
savefig('histogramOfPositiveValuesafterConv5.png')

feat = net.blobs['pool5'].data[0]
plt.subplot(2, 1, 1)
plt.plot(feat.flat)
plt.subplot(2, 1, 2)
_ = plt.hist(feat.flat[feat.flat > 0], bins=100)
savefig('histogramOfPositiveValuesafterPool4.png')

feat = net.blobs['ip2'].data[0,:3]
plt.subplot(2, 1, 1)
plt.plot(feat.flat)
plt.subplot(2, 1, 2)
_ = plt.hist(feat.flat[feat.flat > 0], bins=3)
savefig('histogramOfPositiveValuesafterIP2.png')

feat = net.blobs['prob'].data[0]
plt.plot(feat.flat)
savefig('FinalProbabilityOutput.png')


