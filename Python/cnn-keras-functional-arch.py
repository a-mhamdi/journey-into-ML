#= HOW TO DESIGN A CNN ARCHITECTURE IN KERAS - FUNCTIONAL API =#

# Import the MNIST dataset
from keras.datasets import fashion_mnist
(data_train, labels_train), (data_test, labels_test) = fashion_mnist.load_data()
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'] # fashion_mnist.load_data?

# Normalize to [0, 1]
X_train = data_train/255.0
X_test  = data_test/255.0

# Reshaping data
X_train = X_train.reshape(-1, 28, 28, 1)
X_test  = X_test.reshape(-1, 28, 28, 1)

from keras.utils import to_categorical
y_train = labels_train
y_test = labels_test
	
# Design and build `model` architecture
import keras

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

### ---------------------------------- """ Architecture """ ---------------------------------- ### 
inputs = keras.Input(shape=(28, 28, 1))
# 2-dimensional convolutional layer with 128 feature maps and a 3×3 filter size
L = Conv2D(128, (3, 3), activation='relu')(inputs)
# 2 × 2 max-pooling layer
L = MaxPooling2D((2, 2))(L)
# 2-dimensional convolutional layer with 256 feature maps and a 3×3 filter size
L = Conv2D(256, (3, 3), activation='relu')(L)
# 2 × 2 max-pooling layer
L = MaxPooling2D((2, 2))(L)
# 2-dimensional convolutional layer with 512 feature maps and a 3×3 filter size
L = Conv2D(512, (3, 3), activation='relu')(L)
# 2 × 2 max-pooling layer
L = MaxPooling2D((2, 2))(L)
# flatten the data
L = Flatten()(L)
# dense (fully-connected) layer consisting of 128 units
L = Dense(128, activation='relu')(L)
# dense (fully-connected) layer consisting of 128 units
L = Dense(128, activation='relu')(L)
# Output layer (classification probabilites): 10 neurons
outputs = Dense(10, activation='softmax')(L)
### ------------------------------------------------------------------------------------------- ###
model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_classifier") # Create instance of the graph of layers
model.summary()

keras.utils.plot_model(model, "functional_model_shape_info.png", show_shapes=True)

# Compile `model_` using an appropriate loss and optimizer algorithm
from keras.losses import SparseCategoricalCrossentropy as scc
loss = scc(from_logits=False)
opt = 'adam'
model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])

# Train `model` and assign training meta-data to a variable
mdl_mdata = model.fit(X_train, y_train, validation_split=.2, epochs=8, batch_size=128, shuffle=True)

# Print accuracy of `model` on testing set 
scores = model.evaluate(X_test, y_test, batch_size=32, return_dict=True) # scores['loss'], scores['accuracy']
print("Accuracy: %.2f%%" %(scores['accuracy']*100))

import random
idx = random.randint(0, len(X_test)-1)
sample = X_test[idx, :, :, :].reshape(-1, 28, 28, 1)
y_pred = model.predict(sample)
y_true = y_test[idx]

import matplotlib.pyplot as plt
from numpy import argmax

print('\033[95m', 3*'\t' + 5*'---' + ' Plot an arbitrary sample ' + 5*'---', '\033[0m')
plt.imshow(sample.reshape(28, 28), cmap='gray')
plt.title('Ground Truth: {} | Prediction: {}'.format(classes[argmax(y_true)], classes[argmax(y_pred)]))
plt.axis('off')
plt.show()

# Plot `accuracy` vs `# epoch`
plt.plot(mdl_mdata.history['accuracy'])
plt.plot(mdl_mdata.history['val_accuracy'])
plt.title('CNN Accuracy vs. Epoch')
plt.ylabel('Accuracy')
plt.xlabel('# Epoch')
plt.legend(['Train', 'Test'], loc='best')
plt.show()

# Plot loss vs epoch
plt.plot(mdl_mdata.history['loss'])
plt.plot(mdl_mdata.history['val_loss'])
plt.title('CNN Loss vs. Epoch')
plt.ylabel('Loss')
plt.xlabel('# Epoch')
plt.legend(['Train', 'Test'], loc='best')
plt.show()

model.save('mdls/functional_cnn_model')

