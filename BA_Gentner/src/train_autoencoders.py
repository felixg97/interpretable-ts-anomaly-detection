import pickle
import numpy as np

import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.models import clone_model

# Define the image size
image_size = (28, 28)

def load_data():
    # Load the dataset
    dataset = tfds.load('mnist', split='train', data_dir=r'C:/mnist')

    images, labels = [], []

    # Iterate over the dataset and collect images and labels
    for img_label in dataset:
        image = tf.image.resize(img_label['image'], image_size)
        images.append(image.numpy())
        labels.append(img_label['label'].numpy())

    images = np.array(images)
    labels = np.array(labels)

    return images, labels

images, labels = load_data()

# Flatten and normalize the images
images_flat = images.reshape(images.shape[0], -1)
images_flat = images_flat / 255.0

# Define the autoencoder model
input_dim = images_flat.shape[1]
encoding_dim = 32

input_layer = tf.keras.Input(shape=(input_dim,))
encoder = tf.keras.layers.Dense(128, activation='relu')(input_layer)
encoder = tf.keras.layers.Dense(64, activation='relu')(encoder)
encoder = tf.keras.layers.Dense(32, activation='relu')(encoder)
encoder = tf.keras.layers.Dense(encoding_dim, activation='relu')(encoder)

decoder = tf.keras.layers.Dense(64, activation='relu')(encoder)
decoder = tf.keras.layers.Dense(128, activation='relu')(decoder)
decoder = tf.keras.layers.Dense(input_dim, activation='sigmoid')(decoder)

autoencoder = tf.keras.Model(input_layer, decoder)

def load_clusters(filepath):
    # Load cluster data from a file
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

# Load the previously generated clusters
loaded_cluster_labels = load_clusters('../data/cluster_results_hdbscan.pkl')

cluster_images = {}

for label in range(0, 10):
    # Assign images of each cluster to a dictionary
    cluster_indices = np.where(loaded_cluster_labels == label)[0]
    cluster_images[label] = images_flat[cluster_indices]

autoencoder_models = {}

for label in range(0, 10):
    # Train an autoencoder for each cluster
    autoencoder_models[label] = clone_model(autoencoder)
    autoencoder_models[label].compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder_models[label].fit(cluster_images[label], cluster_images[label], epochs=10, batch_size=128, shuffle=True)

for label in range(0, 10):
    # Save the trained autoencoders
    autoencoder_models[label].save(f'autoencoder_model_{label}.h5')