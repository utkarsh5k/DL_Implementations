#Differences from Theano:
# - stride is the interval at which to apply the convolution
# - unlike theano, we use constant-size input to the network
# - if we don't, we'd need to start swapping
# - the output after convpool is a different size (8,8) here, (5,5) in Theano

#Dataset can be found at http://ufldl.stanford.edu/housenumbers/

#WARNING: Running this will take very long. Over 6 hours. 

#basic imports
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from datetime import datetime

#to load .mat files
from scipy.io import loadmat
from sklearn.utils import shuffle

#Class of the result
def y2indicator(y):
	N = len(y)
	ind = np.zeros((N,10))
	for i in xrange(N):
		ind[i, y[i]] = 1
	return ind

def error_rate(p, t):
	return np.mean(p != t)

def convpool(X, W, b):
	#Set padding as 'same' so that the output is the same size
	conv_output = tf.nn.conv2d(X, W, strides = [1,1,1,1], padding = 'SAME')
	conv_output = tf.nn.bias_add(conv_output, b)
	pool_output = tf.nn.max_pool(conv_output, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
	return pool_output

#Weight Initialisation funtion
def init_filter(shape, poolsize):
	#contains fan-in + fan-out
	w = np.random.randn(*shape) / np.sqrt(np.prod(shape[:-1]) + shape[-1]*np.prod(shape[:-2] / np.prod(poolsize)))
	return w.astype(np.float32)

def rearrange(X):
	#As opposed to theano, here color comes last
	N = X.shape[-1]
	out = np.zeros((N, 32, 32, 3), dtype = np.float32)
	for i in xrange(N):
		for j in xrange(3):
			out[i, :, :, j] = X[:, :, j, i]
	return out / 255

def main():
	train = loadmat('../svhn_cropped/train_32x32.mat')
	test = loadmat('../svhn_cropped/test_32x32.mat')

	#Rearrange to theano format
	Xtrain = rearrange(train)
	
	#Y is a Nx1 matrix of values 1...10 (MATLAB - Indexing)
	#change to 0...9
	Ytrain = train['y'].flatten() - 1
	del train

	Xtrain, Ytrain = shuffle(Xtrain, Ytrain)

	#need a amtrix for cost calculation
	Ytrain_ind = y2indicator(Ytrain)

	Xtest = rearrange(test)
	Ytest = test['y'].flatten() - 1
	del test
	Ytest_ind = y2indicator(Ytest) 

	#intitialise backpropagation parameters
	max_iter = 20 
	print_interval = 10

	learning_rate = np.float32(0.00001)
	reg = np.float32(0.01)
	mom = np.float32(0.99)

	N= Xtrain.shape[0]
	batch_size = 500
	n_batches = N / batch_size

	#Now we want the input size to be constant
	Xtrain = Xtrain[:73000,]
	Ytrain = Ytrain[:73000,]
	Xtest = Xtest[:26000,]
	Ytest = Ytest[:26000]
	Ytest_ind = Ytest_ind[:26000,] 

	#neural net params
	hidden = 500
	num_outputs = 10 
	poolsize = (2,2)

	W1_shape = (5, 5, 3, 20)
	W1_init = init_filter(W1_shape, poolsize)
	b1_init = np.zeros(W1_shape[-1], dtype = np.float32)

	W2_shape = (5, 5, 20, 5 0)
	W2_init = init_filter(W2_shape, poolsize)
	b2_init = np.zeros(W2_shape[-1], dtype = np.float32)

	W3_init = np.random.randn(W2_shape[-1]*8*8, hidden) / np.sqrt(W2_shape*8*8 + hidden)
	b3_init = np.zeros(hidden, dtype = np.float32)
	W4_init = np.random.randn(hidden, num_outputs) / np.sqrt(hidden + num_outputs)
	b4_init = np.zeros(hidden, dtype = np.float32)

	#Now defining tf variables 
	X = tf.placeholder(tf.float32, shape = (batch_size, 32, 32, 3), name = 'X')
	T = tf.placeholder(tf.float32, shape = (batch_size, num_outputs), name = 'T')
	W1 = tf.Variable(W1_init.astype(np.float32))
	b1 = tf.Variable(b1_init.astype(np.float32))
	W2 = tf.Variable(W2_init.astype(np.float32))
	b2 = tf.Variable(b2_init.astype(np.float32))
	W3 = tf.Variable(W3_init.astype(np.float32))
	b3 = tf.Variable(b3_init.astype(np.float32))
	W4 = tf.Variable(W4_init.astype(np.float32))
	b4 = tf.Variable(b4_init.astype(np.float32))

	#feedforward
	Z1 = convpool(X, W1, b1)
	Z2 = convpool(Z1, W2, b2)

	#need to reshape
	Z2_shape = Z2.get_shape().as_list()
	Z2r = tf.reshape(Z2, [Z2_shape[0], np.prod(Z2_shape[1:])])
	Z3 = tf.nn.relu(tf.matmul(Z2r, W3) + b3)
	Yish = tf.matmul(Z3, W4) + b4

	cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(Yish, T))

	train_op = tf.train.RMS_Prop_Optimizer(0.0001, decay = 0.99, momentum = 0.9).minimize(cost)

	#to calculate the error rate
	predict_op = tf.argmax(Yish, 1)

	start_time = datetime.now()
	#This is going to take VERY, VERY LONG. Error might be ugly. Consider running on AWS or on a smaller dataset. 
	LL = []
	init = tf.intitialize_all_variables()
	with tf.Session() as session:
		session.run(init)

		for i in xrange(max_iter):
			for j in xrange(n_batches):
				Xbatch = Xtrain[j*batch_size:(j*batch_size + batch_size),]
				Ybatch = Ytrain_ind[j*batch_size:(j*batch_size + batch_size),]

				if len(Xbatch) == batch_size:
					session.run(train_op, feed_dict = {X: Xbatch, T: Ybatch})
					if j % print_interval == 0:
						#Due to RAM limitations, we need fixed size input
						#cost prediction will be ugly
						test_cost = 0
						prediction = np.zeros(len(Xtest))
						for k in xrange(len(Xtest) / batch_size):
							Xtestbatch = Xtest[k*batch_size: (k*batch_size + batch_size),]
							Ytestbatch = Ytest_ind[k*batch_size: (k*batch_size + batch_size),]
							test_cost += session.run(cost, feed_dict = {X: Xtestbatch, T: Ytestbatch})	
							prediction[k*batch_size: (k*batch_size + batch_size)] = session.run(
								predict_op, feed_dict = {X:Xtestbatch})
						err = error_rate(prediction, Ytest)
						print "Cost / err at iteration i = %d, j = %d: %.3f / %.3f" % (i, j, test_cost, err)
						LL.append(test_cost)
	end_time = datetime.now()
	print "Total time taken: ", (end_time - start_time)		
	plt.plot(LL)
	plt.show()

if __name__ == '__main__':
	main()	