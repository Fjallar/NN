import numpy as np
from scipy.io import loadmat

class NN_layer():

	def __init__(self,N,M,activ_func, nabla=1e-3,beta1=0.9,beta2=0.999):
		#Xavier/Glorot Initialization
		std_dev = np.sqrt(1/(N*M))
		self.W = np.random.randn(N,M)*std_dev
		self.b = np.zeros(M)
		self.forward_func, self.backward_func = activ_func
		self.nabla = nabla
  
		self.moment_W = np.zeros_like(self.W)
		self.moment_b = np.zeros_like(self.b)
		self.rmsprop_W = np.zeros_like(self.W)
		self.rmsprop_b = np.zeros_like(self.b)
		self.t=1

		self.beta1=beta1
		self.beta2=beta2

	def forward(self,n):
		self.n = n
		self.z = np.add(np.matmul(n,self.W),self.b)
		self.np1 = self.forward_func(self.z)
		self.activations = self.backward_func(self.z)

		return self.np1
	
	def update_weights(self, d_W, d_b):
		epsilon = 1e-7
		grad_b=self.nabla*np.mean(d_b,axis=0)
		grad_W=self.nabla*np.mean(d_W,axis=1)
		# Update the first moment estimate (momentum)
		self.moment_W = self.beta1 * self.moment_W + (1 - self.beta1) * grad_W
		self.moment_b = self.beta1 * self.moment_b + (1 - self.beta1) * grad_b

		# Update the second moment estimate (RMSProp)
		self.rmsprop_W = self.beta2 * self.rmsprop_W + (1 - self.beta2) * grad_W ** 2
		self.rmsprop_b = self.beta2 * self.rmsprop_b + (1 - self.beta2) * grad_b ** 2

		# Correct bias in the initial steps
		moment_W_corrct = self.moment_W / (1 - self.beta1 ** self.t)
		moment_b_corrct = self.moment_b / (1 - self.beta1 ** self.t)
		rmsprop_W_corrct = self.rmsprop_W / (1 - self.beta2 ** self.t)
		rmsprop_b_corrct = self.rmsprop_b / (1 - self.beta2 ** self.t)

		# Apply updates
		self.W -= self.nabla * moment_W_corrct / (np.sqrt(rmsprop_W_corrct) + epsilon)
		self.b -= self.nabla * moment_b_corrct / (np.sqrt(rmsprop_b_corrct) + epsilon)
		self.t +=1
	
	def backward(self,grad_np1):
		self.grad_b = grad_np1*self.activations
		self.grad_w = np.transpose(self.n)[:,:,np.newaxis]* self.grad_b[np.newaxis,:,:] # grad_b_i * n_j = grad_w_ij
		self.update_weights(self.grad_w,self.grad_b)

		grad_n = np.matmul(self.grad_b, np.transpose(self.W))
		return grad_n

class NN():
	def __init__(self,layers,nabla=1e-4):
		relu = lambda x: np.maximum(0,x)
		d_relu = lambda x: 0.5*(np.sign(x)+1)
		self.nn_layers = [NN_layer(N,M,(relu,d_relu),nabla=nabla) for N, M in zip(layers[:-1], layers[1:])]

	def forward(self,x):
		for layer in self.nn_layers:
			x=layer.forward(x)
		return x

	def backward(self,d_y):
		for layer in self.nn_layers[::-1]:
			d_y=layer.backward(d_y)
		return d_y

	def evaluate(self, input, label):
		predictions = self.forward(input)
		predicted_classes = np.argmax(predictions,axis=1)
		true_classes = np.argmax(label, axis=1)
		correct = predicted_classes==true_classes
		accuracy = np.sum(correct)/len(predictions)
		return accuracy, correct

class Dataset():
	def __init__(self,file,batch_size=100):
		mnist = loadmat(file)
		mnist_data = mnist["data"].T
		mnist_data = mnist_data/255.0
		mnist_labels = mnist["label"][0]
		mnist_labels = self.one_hot_encode(mnist_labels.astype(int))
		n_train = 60_000 #int(mnist_label.size*0.8)
		self.batch_size = batch_size
		self.train_data, self.test_data, self.train_labels, self.test_labels = (mnist_data[:n_train,:],mnist_data[n_train:,:], mnist_labels[:n_train,:], mnist_labels[n_train:,:])
		
	def one_hot_encode(self,arr):
		# np.eye(10) creates an identity matrix of size 10x10
		# When indexed using arr, it selects the corresponding identity row
		return np.eye(10)[arr]

	def shuffle_data(self):
		indices = np.random.permutation(self.train_labels.shape[0])
		self.train_data=self.train_data[indices,:]
		self.train_labels=self.train_labels[indices,:]
	
	def batch_generator(self):
		num_samples = len(self.train_data)
		indices = np.arange(num_samples)
		np.random.shuffle(indices)

		for start in range(0, num_samples, self.batch_size):
			end = min(start + self.batch_size, num_samples)
			batch_indices = indices[start:end]
			yield self.train_data[batch_indices], self.train_labels[batch_indices]

def print_progress_bar(iteration, total, bar_length=30):
    completed = '=' * int(bar_length * iteration // total)
    remaining = ' ' * (bar_length - len(completed))
    print(f'\rProgress: [{completed}{remaining}] batch: {iteration}/{total}', end='')


if __name__ == "__main__":
	dataset = Dataset("mnist-original",batch_size=32)
	neural_net = NN([784,256,128,64,10],nabla=1e-4)
	epochs=5
 
	for epoch in range(epochs):
		for i, batch in enumerate(dataset.batch_generator()):
			x, t = batch
			y = neural_net.forward(x)
			neural_net.backward(y-t)
			print_progress_bar(i,60_000//32)
		accuracy, _ = neural_net.evaluate(dataset.test_data, dataset.test_labels)
		print()
		print("Epoch {}: Accuracy of {}".format(epoch,accuracy))
		print()
    