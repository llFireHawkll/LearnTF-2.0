from imports import *

"""
    Here we will train a small NN in TF2.0 on fashion MNIST dataset.
    
    STEPS INVOLVED:
        1. Loading Dataset Wrapper
            a. Assigning data in train and test sets
        2. Looking at the shape of datasets
        3. Visualizing the dataset
        4. Normalizing the dataset for NN 
        5. Building simple NN architecture for classification
            a. Build a sequential model
            b. Compile the model
            c. Fit the model on train dataset
            d. Evaluate the model on test dataset

"""

mnist = keras.datasets.fashion_mnist
print(type(mnist))

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print('Train X Size: {0}, Train Y Size: {1}'.format(X_train.shape, y_train.shape))


# Visualizing a sample from train dataset
plt.figure()
plt.imshow(X_train[1])
plt.colorbar()

# Normalizing the dataset for NN
X_train = X_train/255.0
X_test = X_test/255.0

# Simple NN architecture for classification
model = Sequential()
model.add(Flatten(input_shape = (28, 28)))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

# Printing model architecture
print(model.summary())

# Compiling the NN
model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# Fitting the model on train dataset with 10 epochs
model.fit(X_train, y_train, epochs = 10)

# Evaluating the model on test dataset
test_loss, test_acc = model.evaluate(X_test, y_test)
print('\nTest Loss: {0}, Test Accuracy: {1}'.format(test_loss,test_acc))

## Making Prediction on test dataset 
# Here predict_classes() function results the class directly and not softmax values
y_pred = model.predict_classes(X_test)
print('Accuracy on Test Dataset: {0}'.format(accuracy_score(y_test, y_pred)))

# If you want softmax values in predictions
pred = model.predict(X_test)

# After that you need to get argmax to get the class
print('Class for 1 test example: {0}'.format(np.argmax(pred[0])))


