import tensorflow as tf
from keras.api import layers, models 
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# Load the Fashion-MNIST dataset
(x_train, _), (x_test, _) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize the data
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Build the Autoencoder model
input_img = tf.keras.Input(shape=(28, 28, 1))

# Encoder
x = layers.Flatten()(input_img)
encoded = layers.Dense(64, activation='relu')(x)

# Decoder
decoded = layers.Dense(28 * 28, activation='sigmoid')(encoded)
decoded = layers.Reshape((28, 28, 1))(decoded)

# Autoencoder
autoencoder = models.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the Autoencoder
autoencoder.fit(x_train, x_train, epochs=10, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

# Use the encoder to obtain low-dimensional representations
encoder = models.Model(input_img, encoded)
encoded_imgs_train = encoder.predict(x_train)
encoded_imgs_test = encoder.predict(x_test)

# Save the low-dimensional representations of the training and test sets
np.save('encoded_imgs_train.npy', encoded_imgs_train)
np.save('encoded_imgs_test.npy', encoded_imgs_test)

# Use t-SNE for dimensionality reduction and visualization (this is just for display and doesn't affect saved representations)
tsne = TSNE(n_components=2, random_state=42)
x_test_2d = tsne.fit_transform(encoded_imgs_test)

# Visualize the t-SNE results
plt.scatter(x_test_2d[:, 0], x_test_2d[:, 1])
plt.title('t-SNE Visualization of Encoded Images')
plt.show()
