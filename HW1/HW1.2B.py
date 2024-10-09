from tensorflow.keras import layers, models
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load the saved low-dimensional representations
encoded_x_train = np.load('HW1_/encoded_imgs_train.npy')
encoded_x_test = np.load('HW1_/encoded_imgs_test.npy')

# Load the labels from the Fashion-MNIST dataset
(_, y_train), (_, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Build a classifier model (a simple fully connected network)
classifier = models.Sequential([
    layers.InputLayer(input_shape=(64,)),  # The encoded representation is 64-dimensional
    layers.Dense(10, activation='softmax')  # Fashion-MNIST has 10 classes
])

classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the classifier and save the loss and accuracy during the training process
history = classifier.fit(encoded_x_train, y_train, epochs=10, batch_size=256, validation_data=(encoded_x_test, y_test))

# Plot the loss and accuracy curves
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curves')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Curves')
plt.legend()

plt.show()

# Evaluate the model on the test set
test_loss, test_accuracy = classifier.evaluate(encoded_x_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

y_pred = classifier.predict(encoded_x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

correct_indices = np.where(y_pred_classes == y_test)[0]
incorrect_indices = np.where(y_pred_classes != y_test)[0]

# Set the grid size
fig, axes = plt.subplots(2, 5, figsize=(12, 6))  # A grid of 2 rows and 5 columns, first row for correct predictions, second for incorrect

# Display correct predictions
for i, correct in enumerate(correct_indices[:5]):
    ax = axes[0, i]
    ax.imshow(encoded_x_test[correct].reshape(8, 8), cmap='gray')  # Assuming the 64-dimensional encoding can be reshaped to 8x8
    ax.set_title(f"True: {y_test[correct]}\nPred: {y_pred_classes[correct]}")
    ax.axis('off')

# Display incorrect predictions
for i, incorrect in enumerate(incorrect_indices[:5]):
    ax = axes[1, i]
    ax.imshow(encoded_x_test[incorrect].reshape(8, 8), cmap='gray')  # Assuming the 64-dimensional encoding can be reshaped to 8x8
    ax.set_title(f"True: {y_test[incorrect]}\nPred: {y_pred_classes[incorrect]}")
    ax.axis('off')

# Set the overall title
fig.suptitle("Correct Predictions (Top Row) & Incorrect Predictions (Bottom Row)")
plt.subplots_adjust(hspace=0.5)  # hspace controls the row spacing
plt.show()
