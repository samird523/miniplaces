# -*- coding: utf-8 -*-
from DataLoader import *
import resnet_model
import scipy.misc
import os, datetime
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
# Dataset Parameters
batch_size = 128 
load_size = 256
fine_size = 224
c = 3
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])
# Test Parameters
learning_rate = 0.0001
dropout = 0.5 # Dropout, probability to keep units
start_from = '../../../resnet-5000'
# tf Graph input
x = tf.placeholder(tf.float32, [None, fine_size, fine_size, c])
y = tf.placeholder(tf.int64, None)
keep_dropout = tf.placeholder(tf.float32)
train_phase = tf.placeholder(tf.bool)
resnet_size = 34
num_classes = 100
resnet = resnet_model.imagenet_resnet_v2(resnet_size, num_classes)
logits = resnet(x,True)
# define initialization
init = tf.global_variables_initializer()
saver = tf.train.Saver()
testing = "../data/images/test"
classes=100
with tf.Session() as sess:
	if len(start_from)>1:
		saver.restore(sess, start_from)
	else:
		sess.run(init)
	files=os.listdir(testing)
	num_groups=int(len(files)/classes)
	with open("results.txt","w") as result_file:
		for classn in range(classes):
			test_im = []
			filenames = []
			for fileNum in range(num_groups * classn,min(len(files),num_groups*(classn+1))):
				filepath=os.path.join(testing, files[fileNum])
				offset=(load_size-fine_size)/2
				im=scipy.misc.imread(filepath)
				im=scipy.misc.imresize(im, (load_size, load_size))
				im=im.astype(np.float32)/255.
				im=im-np.array(data_mean)
				im=im[offset:offset+fine_size, offset:offset+fine_size, :]
				test_im.append(im)
				filenames.append(files[fileNum])
			val,num = tf.nn.top_k(tf.nn.softmax(logits),k=5,sorted=True)
			test_im = np.array(test_im)
			s, best = sess.run([val, num], feed_dict = {x:test_im, keep_dropout:1.,train_phase:False})
			for done in range(len(filenames)):
				fn = filenames[done]
				vals = " ".join(map(str,best[done]))
				result_file.write("test/"+fn + " "+vals+"\n")