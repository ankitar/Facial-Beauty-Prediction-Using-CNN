import numpy as np
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io
from numpy import *

# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

net = caffe.Classifier('examples/face_feature_extraction/facenet_deploy.prototxt',
                       'examples/face_feature_extraction/facenet_iter_2000.caffemodel')
net.set_phase_test()
net.set_mode_gpu()
# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
#net.set_mean('data', np.load('python/caffe/imagenet/ilsvrc_2012_mean.npy'))  # ImageNet mean
net.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
#net.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
net.set_channel_swap('data', (2,1,0))

list1 = []
list2 = []
#np.corrcoef(list1, list2)
f = open('data/face_feature_extraction/test.txt', 'r')
for line in f:
    line1 = line.split( );
    prediction = net.predict([caffe.io.load_image(line1[0])])
    #print 'prediction shape:', prediction[0].shape 
    #print 'predicted class:', prediction[0]	 
    print 'predicted class:', prediction[0].argmax()
    list2.append(prediction[0].argmax())
    list1.append(int(line1[1]))
    #print list1
    #print list2

print np.corrcoef(list1, list2)
