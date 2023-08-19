#= HOW TO DESIGN A CNN ARCHITECTURE IN KERAS - SEQUENTIAL API =#

from utils.helper_function import get_data

"""
import sys
from importlib import reload
reload(sys.modules["utils.helper_function"])
"""

X_train, X_test, y_train, y_test = get_data()

# Design and build `model_` architecture

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

### ---------------------------------- """ 1st arch """ ---------------------------------- ### 
model = Sequential()
# 2-dimensional convolutional layer with 128 feature maps and a 3×3 filter size
model.add(Conv2D(128, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# 2 × 2 max-pooling layer
model.add(MaxPooling2D((2, 2)))
# 2-dimensional convolutional layer with 256 feature maps and a 3×3 filter size
model.add(Conv2D(256, (3, 3), activation='relu'))
# 2 × 2 max-pooling layer
model.add(MaxPooling2D((2, 2)))
# 2-dimensional convolutional layer with 512 feature maps and a 3×3 filter size
model.add(Conv2D(512, (3, 3), activation='relu'))
# 2 × 2 max-pooling layer
model.add(MaxPooling2D((2, 2)))
# flatten the data
model.add(Flatten())
# dense (fully-connected) layer consisting of 128 units
model.add(Dense(128, activation='relu'))
# dense (fully-connected) layer consisting of 128 units
model.add(Dense(128, activation='relu'))
# Output layer (classification probabilites): 10 neurons
model.add(Dense(10, activation='softmax'))
### ------------------------------------------------------------------------------------------- ###
model_1 = model
model_1.summary()
    
### ---------------------------------- """ 2nd arch """ ---------------------------------- ### 
# 2-dimensional convolutional layer with 128 feature maps and a 3×3 filter size
l1 = Conv2D(128, (3, 3), activation='relu', input_shape=(28, 28, 1))
# 2 × 2 max-pooling layer
l2 = MaxPooling2D((2, 2))
# 2-dimensional convolutional layer with 256 feature maps and a 3×3 filter size
l3 = Conv2D(256, (3, 3), activation='relu')
# 2 × 2 max-pooling layer
l4 = MaxPooling2D((2, 2))
# 2-dimensional convolutional layer with 512 feature maps and a 3×3 filter size
l5 = Conv2D(512, (3, 3), activation='relu')
# 2 × 2 max-pooling layer
l6 = MaxPooling2D((2, 2))
# flatten the data
l7 = Flatten()
# dense (fully-connected) layer consisting of 128 units
l8 = Dense(128, activation='relu')
# dense (fully-connected) layer consisting of 128 units
l9 = Dense(128, activation='relu')
# Output layer (classification probabilites): 10 neurons
l10 = Dense(10, activation='softmax')

mode = Sequential([eval('l'+str(k)) for k in range(1, 11)])
### ------------------------------------------------------------------------------------------- ###
model_2 = model
model_2.summary()

### ---------------------------------- """ 3rd arch """ ---------------------------------- ### 
def my_3rd_arch():
    model = Sequential()
    # 2-dimensional convolutional layer with 128 feature maps and a 3×3 filter size
    model.add(Conv2D(128, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    # 2 × 2 max-pooling layer
    model.add(MaxPooling2D((2, 2)))
    # 2-dimensional convolutional layer with 256 feature maps and a 3×3 filter size
    model.add(Conv2D(256, (3, 3), activation='relu'))
    # 2 × 2 max-pooling layer
    model.add(MaxPooling2D((2, 2)))
    # 2-dimensional convolutional layer with 512 feature maps and a 3×3 filter size
    model.add(Conv2D(512, (3, 3), activation='relu'))
    # 2 × 2 max-pooling layer
    model.add(MaxPooling2D((2, 2)))
    # flatten the data
    model.add(Flatten())
    # dense (fully-connected) layer consisting of 128 units
    model.add(Dense(128, activation='relu'))
    # dense (fully-connected) layer consisting of 128 units
    model.add(Dense(128, activation='relu'))
    # Output layer (classification probabilites): 10 neurons
    model.add(Dense(10, activation='softmax'))

    return model
### ------------------------------------------------------------------------------------------- ###
model_3 = my_3rd_arch() # Create instance of `my_3rd_arch` graph
model_3.summary()

# Compare models
print('`model_1` vs. `model_2`', model_1.get_config() == model_2.get_config())
print('`model_1` vs. `model_3`', model_1.get_config() == model_3.get_config())
print('`model_2` vs. `model_3`', model_2.get_config() == model_3.get_config())

# Compile `model_` using an appropriate loss and optimizer algorithm
loss = 'categorical_crossentropy'
opt = 'adam'
model_3.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

# Train `model_` and assign training meta-data to a variable
cnn_model_mdata = model_3.fit(X_train, y_train, validation_split=.2, epochs=8, batch_size=128, shuffle=True)

import random
idx = random.randint(0, len(X_test)-1)
sample = X_test[idx, :, :, :].reshape(-1, 28, 28, 1)
y_pred = model_3.predict(sample)
y_true = y_test[idx, :]

classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

import matplotlib.pyplot as plt
from numpy import argmax

print('\033[95m', 3*'\t' + 5*'---' + ' Plot an arbitrary sample ' + 5*'---', '\033[0m')
plt.imshow(sample.reshape(28, 28), cmap='gray')
plt.title('Ground Truth: {} | Prediction: {}'.format(classes[argmax(y_true)], classes[argmax(y_pred)]))
plt.axis('off')
plt.show()

# Plot `accuracy` vs `# epoch`
plt.plot(cnn_model_mdata.history['accuracy'])
plt.plot(cnn_model_mdata.history['val_accuracy'])
plt.title('CNN Accuracy vs. Epoch')
plt.ylabel('Accuracy')
plt.xlabel('# Epoch')
plt.legend(['Train', 'Test'], loc='best')
plt.show()

# Plot loss vs epoch
plt.plot(cnn_model_mdata.history['loss'])
plt.plot(cnn_model_mdata.history['val_loss'])
plt.title('CNN Loss vs. Epoch')
plt.ylabel('Loss')
plt.xlabel('# Epoch')
plt.legend(['Train', 'Test'], loc='best')
plt.show()

# Print accuracy of `model_` on testing set 
scores = model_3.evaluate(X_test, y_test)
print("Accuracy: %.2f%%" %(scores[1]*100))

model_3.save('mdls/cnn_model_3')

""" SAVE `model_`
model_1.save('mdls/cnn_model_1')
model_2.save('mdls/cnn_model_2')
model_3.save('mdls/cnn_model_3')
"""
# del model_
""" LOAD `model_`
from keras.models import load_model
model_1 = load_model('mdls/cnn_model_1')
model_2 = load_model('mdls/cnn_model_2')
model_3 = load_model('mdls/cnn_model_3')
"""

