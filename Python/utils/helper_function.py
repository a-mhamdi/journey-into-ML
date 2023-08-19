
def get_data(size_train=60_000, size_test=10_000):
	"""
	Import `MNIST` Dataset.
	
	Arguments:
	----------
		`size_train`: Size of selected portion from train set;
		`size_test`: Size of selected portion from test set.
	
	Returns:
	--------
		Tuple containing train and test sets of both features and targets.
	"""
	# Import the MNIST dataset
	from keras.datasets import fashion_mnist
	(data_train, labels_train), (data_test, labels_test) = fashion_mnist.load_data()

	# Normalize to [0, 1]
	X_train = data_train[0:size_train, :, :]/255.0
	X_test  = data_test[0:size_test, :, :]/255.0
	# Reshaping data
	X_train = X_train.reshape(-1, 28, 28, 1)
	X_test  = X_test.reshape(-1, 28, 28, 1)
	# One Hot Encoding of labels
	from keras.utils import to_categorical
	y_train = to_categorical(labels_train[0:size_train], num_classes=10)
	y_test = to_categorical(labels_test[0:size_test], num_classes=10)
	
	return X_train, X_test, y_train, y_test
	
	
if __name__ == '__main__':
	print('\033[1;92mIMPORTING MNIST FASHION DATA...\033[0;0m', 50*'*', sep='\n') # Used ANSI color code for better rendering
	get_data()
	print(50*'*', '\033[1;92mDONE!\033[0;0m', sep='\n')
