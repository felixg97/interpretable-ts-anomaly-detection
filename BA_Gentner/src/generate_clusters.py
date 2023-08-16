import numpy as np
import pickle

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
import hdbscan

# Define the image size
image_size = (28, 28)

def load_data():
    # Load MNIST dataset from TensorFlow Datasets
    dataset = tfds.load('mnist', split='train', data_dir=r'C:/mnist')

    images, labels = [], []

    # Iterate over the dataset and collect images
    for img_label in dataset:
        image = tf.image.resize(img_label['image'], image_size)
        images.append(image.numpy())

    images = np.array(images)

    return images, labels

images = load_data()

# Flatten and normalize the images
images_flat = images.reshape(images.shape[0], -1)
images_flat = images_flat / 255.0

# Apply PCA
pca = PCA(n_components=0.86, random_state=12)
images_pca = pca.fit_transform(images_flat)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=12)
images_tsne = tsne.fit_transform(images_pca)

# Apply HDBSCAN
np.random.seed(12)
clusterer = hdbscan.HDBSCAN(min_cluster_size=99, prediction_data=True)
cluster_labels = clusterer.fit_predict(images_tsne)

# Plot the clusters
cluster_labels_unique = np.unique(cluster_labels)
for label in cluster_labels_unique:
    mask = cluster_labels == label
    if label == -1:
        plt.scatter(images_tsne[mask, 0], images_tsne[mask, 1], c='black', marker='x', label='Outliers')
    else:
        plt.scatter(images_tsne[mask, 0], images_tsne[mask, 1], label=f'Cluster {label}')

plt.title('MNIST Data - Clusters with Outliers')
plt.legend()
plt.colorbar()
plt.show()

def save_clusters(filepath, data):
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

# Save the clustering results
save_clusters('../data/cluster_results_hdbscan.pkl', cluster_labels)

# Save the images_tsne array
np.save('../data/images_tsne_hdbscan.npy', images_tsne)