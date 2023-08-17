import base64
import copy
from io import BytesIO
import pickle

from dash import dcc
from dash import html
from dash import State, Input, Output
import plotly.graph_objs as go
from matplotlib import pyplot as plt

import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf

def load_data():
    # Load MNIST train dataset from TensorFlow Datasets
    dataset = tfds.load('mnist', split='train', data_dir=r'C:/mnist')

    images = []

    # Iterate over the dataset and collect images
    for img_label in dataset:
        image = img_label['image']
        images.append(image.numpy())

    images = np.array(images)

    return images

def load_clusters(filepath):
    # Load cluster data from a file
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

# Initialize global variables
global cluster_labels
cluster_labels = None

global label_values
label_values = None

global labels_outliers_removed
labels_outliers_removed = None

global images_outliers_removed
images_outliers_removed = None

global selected_cluster
selected_cluster = None

global images_tsne
images_tsne = np.load('../data/images_tsne_hdbscan.npy')

global loaded_cluster_labels
loaded_cluster_labels = load_clusters('../data/cluster_results_hdbscan.pkl')

global images
images = load_data()

global images_flat
# Flatten and normalize the images
images_flat = images.reshape(images.shape[0], -1)
images_flat = images_flat / 255.0

# Convert cluster labels to letters
cluster_labels_unique = np.unique(loaded_cluster_labels)
cluster_names = [chr(ord('A') + label) for label in cluster_labels_unique]

# Plot clusters and outliers
scatter_data = []
for label in cluster_labels_unique:
    mask = loaded_cluster_labels == label
    if label == -1:
        scatter_data.append(go.Scatter(x=images_tsne[mask, 0], y=images_tsne[mask, 1],
                                       mode='markers', marker=dict(color='lightblue', symbol='circle'),
                                       name='Outliers'))
    else:
        scatter_data.append(go.Scatter(x=images_tsne[mask, 0], y=images_tsne[mask, 1],
                                       mode='markers', marker=dict(symbol='circle'),
                                       name=f'Cluster {label}'))

# Create layout for the plot
layout = go.Layout(
    title='MNIST Data - Clusters',
    title_x=0.5,
    showlegend=True,
    width=800,
    height=600
)

# Create the figure with data and layout
global fig
fig = go.Figure(data=scatter_data, layout=layout)

# Create a colormap based on the scatter plot's colorscale
colormap = plt.cm.get_cmap('viridis', len(cluster_labels_unique))

# Update legend entries and colors for existing traces in the figure
for i, (cluster_label, cluster_name) in enumerate(zip(cluster_labels_unique, cluster_names)):
    if cluster_label == -1:  # Skip updating color for outliers
        continue

    color_index = cluster_label / (len(cluster_labels_unique) - 1)
    color_rgb = colormap(color_index)[:3]
    color = f'rgb({int(color_rgb[0] * 255)}, {int(color_rgb[1] * 255)}, {int(color_rgb[2] * 255)})'

    # Find the trace corresponding to the cluster and update its name and marker color
    fig.data[i].name = f"Cluster: {cluster_name}"
    fig.data[i].marker.color = color

def register_callbacks_vial(app):
    @app.callback(
        [Output('scatter-plot', 'figure', allow_duplicate=True),
         Output('mse-vs-num-images', 'figure', allow_duplicate=True)],
        Output('image-frame', 'children', allow_duplicate=True),
        Output('tabs-div', 'style', allow_duplicate=True),
        Input('cluster-dropdown', 'value'),
        prevent_initial_call=True
    )
    # Callback function executed when the cluster-dropdown value is changed
    def cluster_dropdown(cluster):
        # Store the selected cluster in a global variable
        global selected_cluster
        selected_cluster = cluster

        # Initialize MSE plot and image frame as empty values
        mse_vs_num_images_fig = go.Figure()
        image_frame = html.Div()
        tabs_style = {'display': 'none'}

        # Reload the clusters plot based on the selected cluster
        fig_copy = reload_clusters_plot(cluster)

        if cluster is not None:
            # Reload the MSE plot and the image frame based on the selected cluster and display the tabs
            mse_vs_num_images_data = reload_mse_vs_num_images(cluster)
            mse_vs_num_images_fig = go.Figure(data=mse_vs_num_images_data)
            image_frame = reload_image_frame(cluster)
            tabs_style = {'display': 'block'}

        return fig_copy, mse_vs_num_images_fig, image_frame, tabs_style

    @app.callback(
        [Output('scatter-plot', 'figure', allow_duplicate=True),
         Output('mse-vs-num-images', 'figure', allow_duplicate=True)],
        Output('image-frame', 'children', allow_duplicate=True),
        Output('tabs-div', 'style'),
        [Input('mse-vs-num-images', 'clickData')],
        prevent_initial_call=True
    )
    # Callback function executed when the MSE plot is clicked
    def update_plots(click_data):
        # Get the class threshold from the point where the MSE plot is clicked
        class_threshold = click_data['points'][0]['y']

        # Remove outliers from the selected cluster based on the class threshold
        fig_copy = remove_outliers(selected_cluster, class_threshold)

        # Reload the MSE plot and the image frame based on the selected cluster and display the tabs
        mse_vs_num_images_data = reload_mse_vs_num_images(selected_cluster)
        mse_vs_num_images_fig = go.Figure(data=mse_vs_num_images_data)
        image_frame = reload_image_frame(selected_cluster)
        tabs_style = {'display': 'block'}

        return fig_copy, mse_vs_num_images_fig, image_frame, tabs_style

    @app.callback(
        [Output('image-frame', 'children'), Output('scatter-plot', 'figure'), Output('mse-vs-num-images', 'figure')],
        Input('delete-button', 'n_clicks'),
        State('image-frame', 'children'),
        prevent_initial_call=True
    )
    # Callback function executed when the delete button is clicked
    def update_image_frame(n_clicks, image_elements_st):
        global images_tsne, loaded_cluster_labels, images, images_flat

        image_elements = []

        # Get the indices of the images to delete
        images_to_delete_indices = []
        if image_elements_st is not None:
            for i, element in enumerate(image_elements_st):
                checkbox = element.get('props', {}).get('children', [])[0]
                if checkbox.get('props', {}).get('value') == ['checked']:
                    image_id = element['props']['id']
                    image_index = int(image_id.split('-')[1])
                    images_to_delete_indices.append(image_index)

        # Delete the selected images
        if images_to_delete_indices:
            cluster_indices = np.where(loaded_cluster_labels != 999999999)[0]
            cluster_indices = np.delete(cluster_indices, images_to_delete_indices)

            # Remove the deleted images from the global variables
            images_tsne = images_tsne[cluster_indices]
            images = images[cluster_indices]

            # Flatten and normalize the images
            images_flat = images.reshape(images.shape[0], -1)
            images_flat = images_flat / 255.0

            loaded_cluster_labels = loaded_cluster_labels[cluster_indices]

        # Reload clusters plot and MSE plot without deleted images
        fig = reload_clusters_plot(selected_cluster)
        mse_vs_num_images_data = reload_mse_vs_num_images(selected_cluster)
        mse_vs_num_images_fig = go.Figure(data=mse_vs_num_images_data)

        # Reload image frame without deleted images
        if selected_cluster is not None:
            image_elements = reload_image_frame(selected_cluster)

        return image_elements, fig, mse_vs_num_images_fig

    @app.callback(
        Output('images-labels', 'children'),
        [Input('tabs', 'value')]
    )
    # Callback function executed when the tab is changed
    def reload_label_assignment(tabs_value):
        images_html_list = []

        for index, selected_label in enumerate(cluster_labels_unique[1:]):
            # Extract five images belonging to the selected cluster
            cluster_indices = np.where(loaded_cluster_labels == selected_label)[0]
            images_selected = images[cluster_indices]
            images_selected = images_selected[:5]

            # Show values of previously assigned labels
            if label_values is None:
                label_value = None
            else:
                label_value = label_values[index]

            # Create a Div element containing the images and a Dropdown for the cluster label
            images_html = html.Div(style={'display': 'flex', 'flexDirection': 'row', 'alignItems': 'center'}, children=[
                html.Div(
                    [html.Img(src=f"data:image/png;base64,{get_encoded_image(img)}", style={"margin": "5px", "width": '50px',
                        "height":'50px'}) for img in images_selected], style={"margin": "10px 0px -5px 10px"}
                ),
                dcc.Dropdown(
                    id=f'label-dropdown {index}',
                    options=[
                        {'label': label, 'value': label}
                        for label in cluster_labels_unique[1:]
                    ],
                    value=label_value,
                    placeholder="Select a label",
                    style={'width': '200px', "margin": "8px 0px 0px 5px"},
                ),
            ])
            images_html_list.append(images_html)

        return images_html_list

    @app.callback(
        Output('safe-info', 'children'),
        [Input('save-labels-button', 'n_clicks')],
        State('label-dropdown 0', 'value'),
        State('label-dropdown 1', 'value'),
        State('label-dropdown 2', 'value'),
        State('label-dropdown 3', 'value'),
        State('label-dropdown 4', 'value'),
        State('label-dropdown 5', 'value'),
        State('label-dropdown 6', 'value'),
        State('label-dropdown 7', 'value'),
        State('label-dropdown 8', 'value'),
        State('label-dropdown 9', 'value'),
        prevent_initial_call=True
    )
    # Callback function executed when the save-labels-button is clicked
    def safe_labels(clicks, label_0, label_1, label_2, label_3, label_4, label_5, label_6, label_7, label_8, label_9):
        label_variables = [label_0, label_1, label_2, label_3, label_4, label_5, label_6, label_7, label_8, label_9]

        # Safe assigned labels in a global variable
        global label_values
        label_values = label_variables

        for value_to_check in range(10):
            # Count the occurrences of the value in the label variables
            count_value = sum(label_var == value_to_check for label_var in label_variables)

            if count_value > 1:
                # Display a warning message if one label is assigned to multiple clusters
                return html.Div(
                    children=[
                        html.Div("Please select a unique label for each cluster!", style={'color': 'red', 'margin': '-5px 0px 15px 15px'}),
                    ]
                )

        if label_0 is None or label_1 is None or label_2 is None or label_3 is None or label_4 is None or label_5 is None or label_6 is None or label_7 is None or label_8 is None or label_9 is None:
            # Display a warning message if any cluster has not been labeled
            return html.Div(
                children=[
                    html.Div("Please select a label for each cluster!", style={'color': 'red', 'margin': '-5px 0px 15px 15px'}),
                ]
            )
        else:
            # Define a mapping from integer values to letters
            value_to_letter = {-1: 'Outlier', 0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J'}

            # Replace the integer values in cluster_labels_mapped with letters
            cluster_labels_letters = [value_to_letter[value] for value in loaded_cluster_labels]

            for label_var, letter in zip(label_variables, cluster_names[1:]):
                # Replace the letters with the assigned labels
                cluster_labels_letters = [label_var if label == letter else label for label in cluster_labels_letters]

            # Remove outliers and save the labels and images in global variables for FSL
            global labels_outliers_removed, images_outliers_removed
            cluster_labels_letters = np.array(cluster_labels_letters)
            mask = cluster_labels_letters != 'Outlier'
            labels_outliers_removed = cluster_labels_letters[mask].astype(float)
            images_outliers_removed = images[mask]

            return html.Div(
                # Return a success message if the labels have been saved successfully
                children=[
                    html.Div("Your labels have been saved!", style={'color': 'green', 'margin': '-5px 0px 15px 15px'}),
                ]
            )

def calculate_mse_vs_num_images(cluster_label):
    # Load the saved autoencoder
    autoencoder = tf.keras.models.load_model(f'../autoencoder_models/autoencoder_model_{cluster_label}.h5')
    cluster_indices = np.where(loaded_cluster_labels == cluster_label)[0]

    # Calculate the MSE for each sample
    mse_class = calculate_mse(images_flat[cluster_indices], autoencoder)
    num_instances = len(cluster_indices)
    num_images = list(range(num_instances))

    return np.sort(mse_class), num_images

def calculate_mse(images, autoencoder_model):
    # Use the autoencoder model to reconstruct the input images
    reconstructed_images = autoencoder_model.predict(images)

    # Calculate the MSE between the original images and their reconstructions
    mse = np.mean((images - reconstructed_images) ** 2, axis=1)
    return mse

def remove_outliers(selected_cluster, class_threshold):
    global images_tsne, loaded_cluster_labels, fig, images, images_flat

    # Convert selected cluster label to numeric index
    selected_label = ord(selected_cluster) - ord('A')

    # Load the autoencoder model for the selected cluster
    autoencoder = tf.keras.models.load_model(f'../autoencoder_models/autoencoder_model_{selected_label}.h5')

    # Get the indices of data points in the selected cluster
    cluster_indices = np.where(loaded_cluster_labels == selected_label)[0]

    # Calculate the MSE for each sample in the selected cluster
    mse_class = calculate_mse(images_flat[cluster_indices], autoencoder)

    # Create a mask to identify outliers in the selected cluster based on the class_threshold
    outlier_mask = np.zeros_like(loaded_cluster_labels, dtype=bool)
    class_outlier_mask = mse_class > class_threshold
    outlier_mask[cluster_indices] = class_outlier_mask

    # Get the indices of data points that are not outliers in the selected cluster
    clean_indices = np.where(~outlier_mask)[0]

    # Remove outliers from global variables
    clean_cluster_labels = loaded_cluster_labels[clean_indices]
    images_tsne = images_tsne[clean_indices]
    loaded_cluster_labels = clean_cluster_labels
    images = images[clean_indices]

    # Flatten and normalize the images
    images_flat = images.reshape(images.shape[0], -1)
    images_flat = images_flat / 255.0

    # Reload the clusters plot and highlight the selected cluster
    fig = reload_clusters_plot(selected_cluster)

    return fig

def reload_clusters_plot(selected_cluster):
    # Plot clusters and outliers
    scatter_data_dicts = []
    for label in cluster_labels_unique:
        if label == -1:
            mask = loaded_cluster_labels == -1
            scatter_data_dicts.append(
                dict(
                    x=images_tsne[mask, 0],
                    y=images_tsne[mask, 1],
                    mode='markers',
                    marker=dict(color='lightblue', symbol='circle'),
                    name='Outliers'
                )
            )
        else:
            mask = loaded_cluster_labels == label
            scatter_data_dicts.append(
                dict(
                    x=images_tsne[mask, 0],
                    y=images_tsne[mask, 1],
                    mode='markers',
                    marker=dict(symbol='circle'),
                    name=f'Cluster {label}'
                )
            )

    # Create a copy of the original figure
    fig_copy = copy.deepcopy(fig)

    # Update traces with the new scatter_data
    for i, trace_dict in enumerate(scatter_data_dicts):
        fig_copy.data[i].update(trace_dict)

    # Update legend entries and colors for existing traces in the figure
    for i, (cluster_label, cluster_name) in enumerate(zip(cluster_labels_unique, cluster_names)):
        if cluster_label == -1:  # Skip updating color for outliers
            continue

        color_index = cluster_label / (len(cluster_labels_unique) - 1)
        color_rgb = colormap(color_index)[:3]  # Get RGB values from the colormap
        color = f'rgb({int(color_rgb[0] * 255)}, {int(color_rgb[1] * 255)}, {int(color_rgb[2] * 255)})'

        # Find the trace corresponding to the cluster and update its name and marker color
        fig_copy.data[i].name = f"Cluster: {cluster_name}"
        fig_copy.data[i].marker.color = color

        # Set opacity of traces based on whether they are the selected cluster
        if cluster_name == selected_cluster or selected_cluster is None:
            fig_copy.data[i].marker.opacity = 1
        else:
            fig_copy.data[i].marker.opacity = 0.1

    return fig_copy

def reload_mse_vs_num_images(selected_cluster):
    # Convert selected cluster label to numeric index
    selected_label = ord(selected_cluster) - ord('A')

    # Calculate the MSE and corresponding number of images for the selected cluster
    mse_values, num_images = calculate_mse_vs_num_images(selected_label)

    # Create 'MSE vs. Number of Images' plot
    mse_vs_num_images_data = [
        go.Scatter(
            x=list(num_images),
            y=mse_values,
            mode='lines',
            name=f'Cluster {selected_cluster}',
            line=dict(color='blue', width=2)
        )
    ]
    # Define layout of 'MSE vs. Number of Images' plot
    layout = go.Layout(
        title='MSE vs. Number of Images',
        title_x=0.5,
        showlegend=False,
        xaxis=dict(title='Number of Images'),
        yaxis=dict(title='Mean Squared Error')
    )

    return go.Figure(data=mse_vs_num_images_data, layout=layout)

def reload_image_frame(selected_cluster):
    global loaded_cluster_labels, images

    # Convert selected cluster label to numeric index
    selected_label = ord(selected_cluster) - ord('A')

    # Get the images of the selected cluster
    cluster_indices = np.where(loaded_cluster_labels == selected_label)[0]
    images_selected = images[cluster_indices]

    # Flatten and normalize the images
    images_selected_flat = images_selected.reshape(images_selected.shape[0], -1)
    images_selected_flat = images_selected_flat / 255.0

    # Calculate the MSE values of the selected images
    mse_images_selected = calculate_mse(images_selected_flat, tf.keras.models.load_model(f'../autoencoder_models/autoencoder_model_{selected_label}.h5'))
    sorted_indices = np.argsort(mse_images_selected)[::-1]

    # Sort the selected images based on MSE in descending order
    images_selected = [images_selected[i] for i in sorted_indices]
    image_elements = []

    for i, img in enumerate(images_selected):
        img_id = f'image-{cluster_indices[sorted_indices[i]]}'
        checkbox_id = f'checkbox-{cluster_indices[sorted_indices[i]]}'

        # Create checkbox and image elements
        checkbox = dcc.Checklist(
            id=checkbox_id,
            options=[{'label': '', 'value': 'checked'}],
            value=[],
            style={'margin': '5px', 'float': 'left'}
        )
        img_element = html.Img(
            src='data:image/png;base64,{}'.format(get_encoded_image(img)),
            width='80px',
            height='80px',
            style={'cursor': 'pointer', 'margin': '5px'}
        )

        # Create a div element containing the checkbox and image elements
        image_element = html.Div(
            id=img_id,
            children=[
                checkbox,
                img_element
            ],
            **{'data-index': i},
        )
        image_elements.append(image_element)

    return image_elements

def get_encoded_image(image):
    # Convert image to base64 encoded string
    image_pil = tf.keras.preprocessing.image.array_to_img(image)
    image_buffer = BytesIO()
    image_pil.save(image_buffer, format='PNG')
    encoded_image = base64.b64encode(image_buffer.getvalue()).decode('utf-8')
    return encoded_image