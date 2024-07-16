# import pickle
# from sklearn.datasets import fetch_openml
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split

# X, y = fetch_openml(name='mnist_784', version=1, return_X_y=True)


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# clf= RandomForestClassifier(n_jobs=1)

# clf.fit(X_train, y_train)


# print(clf.score(X_test, y_test))

# with open('mnist_model.pkl', 'wb') as f:
#     pickle.dump(clf, f)








import pickle
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Fetch the MNIST dataset
X, y = fetch_openml(name='mnist_784', version=1, return_X_y=True)

# Convert the DataFrame to a NumPy array
X = X.values

# Convert the target labels to a one-hot encoded format
y = to_categorical(y.astype(int))

# Reshape the data to fit the CNN input
X = X.reshape(-1, 28, 28, 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create the CNN model
model = Sequential()
# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

model.add(Conv2D(128, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on the test set
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Save the model
with open('mnist_cnn_model.pkl', 'wb') as f:
    pickle.dump(model, f)
