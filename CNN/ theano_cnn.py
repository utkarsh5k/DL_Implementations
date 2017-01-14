#An implementation of Convolution Neural Network in Theano. This will be tested on SVHN dataset.
#Dataset can be found at http://ufldl.stanford.edu/housenumbers/

#basic imports
import numpy as np
import theano
import theano.tensor as tnr

#graphing
import matplotlib.pyplot as plt

#imports for convolution and max-pool layers
from theano.tensor.nnet import conv2d
from theano.tensor.signal import downsample 

from scipy.io import loadmat #to load the .mat files
from sklearn.utils import shuffle

from datetime import datetime

def error_rate(p, t):
	return np.mean(p != t)

def relu(a):
	return a * (a > 0)
#Relu is simply max(a,0)

#Class of the result
def y2indicator(y):
	N = len(y)
	ind = np.zeros((N,10))
	for i in xrange(N):
		ind[i, y[i]] = 1
	return ind

#convpool layer. The convolution and pooling aren't seperate, because an alternation fits well. 
#Pool size is almost always (2,2)
def convpool(X, W, b, poolsize=(2,2)):
	conv_output = conv2d(input = X, filters = W) 
	pooling_output = downsample.max_pool_2d(
		input = conv_output,
		ds = poolsize,
		ignore_border = True
		)
	# add the bias term. Since the bias is a vector (1D array), we first
    # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
    # thus be broadcasted across mini-batches and feature map
    # width & height
    # return T.tanh(pooled_out + b.dimshuffle('x', 0, 'x', 'x'))
	return relu(pooled_out + b.dimshuffle('x', 0, 'x', 'x'))

#Weight Initialisation funtion
def init_filter(shape, poolsize):
	#contains fan-in + fan-out
	w = np.random.randn(*shape) / np.sqrt(np.prod(shape[1:]) + shape[0]*np.prod(shape[2:] / np.prod(poolsize)))
	return w.astype(np.float32)

#New rearrange function beacaue theano expects Nx3x32x32 image
def rearrange(X):
	N = X.shape[-1]
	out = np.zeros((N, 3, 32, 32), dtype = np.float32)
	for i in xrange(N):
		for j in xrange(3):
			out[i, j, :, :] = X[:, :, j, i]
	return out/255

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

	#Last stage fc-nn has 500 hidden stages, 10 outputs
	hidden = 500
	num_outputs = 10

	poolsize = (2,2)

	W1_shape = (20, 3, 5, 5)
	W1_init = init_filter(W1_shape, poolsize)
	b1_init = np.zeros(W1_shape[0] dtype = np.float32)

	#Important: 2nd element of W2_shape should be equal to 
	#           first  element of W1_shape
	W2_shape = (50, 20, 5, 5)
	W2_init = init_filter(W1_shape, poolsize)
	b2_init = np.zeros(W2_shape[0], dtype = np.float32)

	#vanilla neural network 
	W3_init = np.random.randn(W2_shape[0]*5*5, hidden) / np.sqrt(W2_shape[0]*5*5 + hidden)
	b3_init = np.zeros(M, dtype = np.float32)

	#last set of weights
	W4_init = np.random.randn(hidden, num_outputs) / np.sqrt(hidden + num_outputs)
	b4_init = np.zeros(K, dtype = np.float32)

	#Defining the theano output
	#X is a 4D tensor
	X = tnr.tensor4('X', dtype = 'float32')
	Y = tnr.matrix('T')
	W1 = theano.shared(W1_init, 'W1') 
	b1 = theano.shared(b1_init, 'b1')
	W2 = theano.shared(W1_init, 'W2')
	b2 = theano.shared(b1_init, 'b2')
	W3 = theano.shared(W1_init.astype(np.float32), 'W3')
	b3 = theano.shared(b1_init, 'b3')
	W4 = theano.shared(W1_init.astype(np.float32), 'W4')
	b4 = theano.shared(b1_init, 'b4')

	#Using momentum, so we need it as theano variable
	dW1 = theano.shared(np.zeros(W1_init.shape, dtype = np.float32), 'dW1')
	db1 = theano.shared(np.zeros(b1_init.shape, dtype = np.float32), 'db1')
	dW2 = theano.shared(np.zeros(W2_init.shape, dtype = np.float32), 'dW2')
	db2 = theano.shared(np.zeros(b2_init.shape, dtype = np.float32), 'db2')
	dW3 = theano.shared(np.zeros(W3_init.shape, dtype = np.float32), 'dW3')
	db3 = theano.shared(np.zeros(b3_init.shape, dtype = np.float32), 'db3')
	dW4 = theano.shared(np.zeros(W4_init.shape, dtype = np.float32), 'dW4')
	db4 = theano.shared(np.zeros(b4_init.shape, dtype = np.float32), 'db4')

	#Forward-pass
	Z1 = convpool(X, W1, b1)
	Z2 = convpool(Z1, W2, b2)
	Z3 = relu(Z2.flatten(ndim=2).dot(W3) + b3)
	pY = tnr.nnet.softmax(Z3.dot(W4) + b4)

	#Cost calculation
	#Using regulatization 
	params = (W1, b1, W2, b2, W3, b3, W4, b4)
	reg_cost = reg*np.sum((param*param).sum() for param in params)
	cost = -(Y * tnr.log(pY)).sum() + reg_cost
	prediction = tnr.argmax(pY, axis = 1)

	#updating values
	update_W1 = W1 + mom*dW1 - learning_rate*tnr.grad(cost, W1)
	update_b1 = b1 + mom*db1 - learning_rate*tnr.grad(cost, b1)
	update_W2 = W2 + mom*dW2 - learning_rate*tnr.grad(cost, W2)
	update_b2 = b2 + mom*db2 - learning_rate*tnr.grad(cost, b2)
	update_W3 = W3 + mom*dW3 - learning_rate*tnr.grad(cost, W3)
	update_b3 = b3 + mom*db3 - learning_rate*tnr.grad(cost, b3)
	update_W4 = W4 + mom*dW4 - learning_rate*tnr.grad(cost, W4)
	update_b4 = b4 + mom*db4 - learning_rate*tnr.grad(cost, b4)

	update_dW1 = mom*dW1 - learning_rate*tnr.grad(cost, W1)
	update_db1 = mom*db1 - learning_rate*tnr.grad(cost, b1)
	update_dW2 = mom*dW2 - learning_rate*tnr.grad(cost, W2)
	update_db2 = mom*db2 - learning_rate*tnr.grad(cost, b2)
	update_dW3 = mom*dW3 - learning_rate*tnr.grad(cost, W3)
	update_db3 = mom*db3 - learning_rate*tnr.grad(cost, b3)
	update_dW4 = mom*dW4 - learning_rate*tnr.grad(cost, W4)
	update_db4 = mom*db4 - learning_rate*tnr.grad(cost, b4)

	#time to train (and take a nap for hours)
	train = theano.function(
		inputs = [X, Y],
		updates = [
			(W1, update_W1),
			(b1, update_b1),
			(W2, update_W2),
			(b2, update_b2),
			(W3, update_W3),
			(b3, update_b3),
			(W4, update_W4),
			(b4, update_b4),
			(dW1, update_dW1),
			(db1, update_db1),
			(dW2, update_dW2),
			(db2, update_db2),
			(dW3, update_dW3),
			(db3, update_db3),
			(dW4, update_dW4),
			(db4, update_db4),
		]
	)

	get_prediction = theano.function(
		inputs = [X,Y],
		outputs = [cost, prediction],
	)

	#actual training. Took me over 3 hours. You can go sleep now. 
	LL = []
	start_time = datetime.now()
	for i in xrange(max_iter):
		for j in xrange(n_batches):
			Xbatch = Xtrain[j*batch_size:(j*batch_size + batch_size),]
			Ybatch = Ytrain_ind[j*batch_size:(j*batch_size + batch_size),]

			train(Xbatch, Ybatch)
			if j % print_interval == 0:
				cost_val, prediction_val = get_prediction(Xtest, Ytest_ind)
				err = error_rate(prediction_val, Ytest)
				print "Cost / error at iteration i = %d, j = %d: %.3f / %.3f" % (i, j, cost_val, err)	
				LL.append(cost_val)
	end_time = datetime.now()
	print "Total time taken: ", (end_time - start_time)
	plt.show(LL)
	plt.show()

	#visualising the filters. Comment out if you unnecessary 
   	#Filter 1: (20, 3, 5, 5)
	W1_weights = W1.get_values()
	grid = np.zeros((8*5, 8*5))

	#m, n tell the location where we want to draw on the grid
	m = 0
	n = 0
	for i in xrange(20):
		for j in xrange(3):
			filt = W1_weights[i,j]
			grid[m*5:(m+1)*5, n*5:(n+1)*5] = filt
			m += 1
			if m >= 8:
				m = 0
				n += 1
	plt.imshow(grid, cmap = 'gray')
	plt.title("W1")
	plt.show()

	#Filter 2: (50, 20, 5, 5)
	W2_weights = W2.get_values()
	grid = np.zeros((32*5, 32*5))

	#m, n tell the location where we want to draw on the grid
	m = 0
	n = 0
	for i in xrange(50):
		for j in xrange(20):
			filt = W2_weights[i,j]
			grid[m*5:(m+1)*5, n*5:(n+1)*5] = filt
			m += 1
			if m >= 32:
				m = 0
				n += 1
	plt.imshow(grid, cmap = 'gray')
	plt.title("W2")
	plt.show()


if __name__ == '__main__':
		main()
		


