import random

import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State, ALL

import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense, Layer
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model

import VIAL
import XAI

# Initialize global variables
global test_images_exp
test_images_exp = None

global test_labels_exp
test_labels_exp = None

layers = []

# Define layer options for dropdown
layer_options = [
    {"label": "Conv2D", "value": "Conv2D"},
    {"label": "Dropout", "value": "Dropout"},
    {"label": "MaxPooling2D", "value": "MaxPooling2D"},
    {"label": "Flatten", "value": "Flatten"},
    {"label": "Dense", "value": "Dense"}
]

def register_callbacks_fsl(app):
    @app.callback(
        Output("layer-container", "children"),
        Output("total-layers", "children"),
        Input("add-layer-button", "n_clicks"),
        Input({"type": "delete-button", "index": ALL}, "n_clicks"),
        Input({"type": "layer-type", "index": ALL}, "value"),
        Input({"type": "up-button", "index": ALL}, "n_clicks"),
        Input({"type": "down-button", "index": ALL}, "n_clicks"),
        State("layer-container", "children"),
        State("total-layers", "children"),
        prevent_initial_call=True
    )
    # Callback function executed when adding, deleting, or reordering custom layers
    def update_layer_container(n_clicks, delete_clicks, layer_types, up_clicks, down_clicks,
                               existing_layers, total_layers):

        # Get the ID of the component that triggered the callback
        ctx = dash.callback_context
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

        layers = existing_layers or []
        total_layers = len(layers)

        # Add a new layer if the "Add Layer" button was clicked
        if triggered_id == "add-layer-button":
            new_layer_index = len(layers)
            layer = create_layer_div(new_layer_index)
            layers.append(layer)
            total_layers += 1

        # Remove the corresponding layer if any delete button was clicked
        elif any("delete-button" in t["prop_id"] for t in ctx.triggered):
            delete_button_ids = [int(t["prop_id"].split(",")[-2].split(".")[0].split(":")[1]) for t in ctx.triggered
                                 if "delete-button" in t["prop_id"]]
            layers = [layer for layer in layers if int(layer["props"]["id"]["index"]) not in delete_button_ids]
            total_layers -= len(delete_button_ids)

        # Update the layer details if any layer type dropdown was changed
        elif any("layer-type" in t["prop_id"] for t in ctx.triggered):
            modified_layers = []
            for i, layer in enumerate(layers):
                layer_type = layer_types[i]
                details_id = {"type": "layer-details", "index": layer["props"]["id"]["index"]}
                layer_details = get_layer_details(layer_type, layer["props"]["id"]["index"])
                layer["props"]["children"][1] = html.Div(id=details_id, children=layer_details)
                modified_layers.append(layer)
            layers = modified_layers

        # Move the corresponding layer up if any up button was clicked
        elif any("up-button" in t["prop_id"] for t in ctx.triggered):
            up_button_ids = [int(t["prop_id"].split(",")[-2].split(".")[0].split(":")[1]) for t in ctx.triggered
                             if "up-button" in t["prop_id"]]
            for i, layer in enumerate(layers):
                layer_index = int(layer["props"]["id"]["index"])
                if layer_index in up_button_ids:
                    if i > 0:
                        layers[i], layers[i - 1] = layers[i - 1], layers[i]
                    else:
                        layers[i], layers[total_layers - 1] = layers[total_layers - 1], layers[i]
                    break

        # Move the corresponding layer down if any down button was clicked
        elif any("down-button" in t["prop_id"] for t in ctx.triggered):
            down_button_ids = [int(t["prop_id"].split(",")[-2].split(".")[0].split(":")[1]) for t in ctx.triggered
                               if "down-button" in t["prop_id"]]
            for i, layer in enumerate(layers):
                layer_index = int(layer["props"]["id"]["index"])
                if layer_index in down_button_ids:
                    if i < total_layers - 1:
                        layers[i], layers[i + 1] = layers[i + 1], layers[i]
                    else:
                        layers[i], layers[0] = layers[0], layers[i]
                    break

        return layers, total_layers

    @app.callback(
        Output("loading-training-results", "fullscreen"),
        Output("training-results", "children"),
        Output("alert-output", "children"),
        Output("alert-no-training", "children"),
        Output('cm-div', 'children'),
        Output('alert-no-explanation', 'children'),
        Input("train-button", "n_clicks"),
        State("model-dropdown", "value"),
        State("layers-dropdown", "value"),
        State('num-episodes-input', 'value'),
        State('num-inner-updates-input', 'value'),
        State('num-epochs-input', 'value'),
        State('learning-rate-input', 'value'),
        State('batch-size-input', 'value'),
        State("layer-container", "children"),
        State("data-augmentation-checklist", 'value')
    )
    # Callback function executed when the "Train Model" button is clicked
    def train_model(n_clicks, selected_model, selected_layers, num_episodes, num_inner_updates, num_epochs,
                    learning_rate, batch_size, layers, data_augmentation):
        is_loading = n_clicks is not None
        pretrained_model = False

        alert_explanation = dbc.Alert(f"Please train the model", dismissable=False,
                                      color="info")

        # Return alerts if the "Train Model" button has not been clicked yet
        if n_clicks is None:
            alert_training = dbc.Alert(f"Please train the model", dismissable=False, color="info")
            return True, [], [], alert_training, html.Div(), alert_explanation

        try:
            # Raise error if labels are not assigned
            if VIAL.labels_outliers_removed is None:
                raise ValueError(f"Please assign labels before training the model")

            # Configure layers based on user input if the selected layers are custom
            if selected_layers == "custom":
                all_layers = []
                num = 0
                for index, layer in enumerate(layers):
                    num += 1
                    layer_type = layer["props"]["children"][0]["props"]["value"]
                    layer_details = layer["props"]["children"][1]["props"]["children"]

                    if num == 1:
                        # Add input shape details to the first layer
                        input_shape_label = {'props': {'children': 'Input Shape:'}, 'type': 'Label',
                                             'namespace': 'dash_html_components'}
                        input_shape_value = {'props': {'value': 'input_shape=(28, 28, 3)', 'type': 'text',
                                                       'id': {'type': 'input-shape', 'index': 0}}, 'type': 'Input',
                                             'namespace': 'dash_core_components'}
                        layer_details = [layer_details] + [input_shape_label, input_shape_value]

                    layer_code = generate_layer_code(layer_type,layer_details)
                    all_layers.append(layer_code)

                # Convert layers to a Keras model
                final_model = create_keras_model(all_layers)

                # Get the last layer of the model
                last_layer = final_model.layers[-1]

                # Raise error if the last layer does not have 10 units
                if last_layer.units != 10:
                    raise ValueError(
                        f"The last layer of the model must have 10 units, but it has {last_layer.units} units")

            # Create a pretrained model if the selected layers are pretrained
            elif selected_layers == "pretrained":
                final_model = create_pretrained_model()
                pretrained_model = True

            # Train MAML model if the selected model is MAML
            if selected_model == 'maml':
                results = maml(final_model, num_inner_updates, learning_rate, batch_size, num_episodes,
                               data_augmentation, pretrained_model)

            # Train regular model if the selected model is regular
            elif selected_model == 'regular':
                results = regular_model(final_model, batch_size, num_epochs, learning_rate, data_augmentation, pretrained_model)

            return is_loading, results, [], html.Div(), XAI.load_cm(final_model, test_images_exp,
                                                                    test_labels_exp), html.Div()

        # Display an alert with the error message if an error occurs during training
        except Exception as e:
            alert = dbc.Alert(f"An error occurred: {e}", dismissable=False, color="warning")

            return is_loading, [], alert, html.Div(), html.Div(), alert_explanation

    @app.callback(
        Output("num-episodes-div", "style"),
        Output("num-inner-updates-div", "style"),
        Output("num-epochs-div", "style"),
        Output("batch-size-input", "value"),
        Input("model-dropdown", "value"),
    )
    # Callback function executed when the model selection dropdown is changed
    def show_hide_input_fields(selected_model):
        if selected_model == None:
            return None

        # Show the relevant input fields for MAML and hide others if the selected model is MAML
        if selected_model == 'maml':
            episodes_div_style = {'display': 'block', "margin": "20px 15px"}
            inner_updates_div_style = {'display': 'block', "margin": "20px 15px"}
            epochs_div_style = {'display': 'none'}
            batch_size_input = 32

        # Show the relevant input fields for the regular model and hide others if the selected model is regular
        elif selected_model == 'regular':
            episodes_div_style = {'display': 'none'}
            inner_updates_div_style = {'display': 'none'}
            epochs_div_style = {'display': 'block', "margin": "20px 15px"}
            batch_size_input = 256

        return episodes_div_style, inner_updates_div_style, epochs_div_style, batch_size_input

    @app.callback(
        Output("layer-div", "style"),
        Input("layers-dropdown", "value"),
    )
    # Callback function executed when the layers dropdown is changed
    def show_hide_layer_fields(selected_layer):
        if selected_layer == None:
            return None

        # Show custom layer configuration if the selected layer is custom
        if selected_layer == 'custom':
            layer_div_style = {'display': 'block'}

        # Hide custom layer configuration if the selected layer is pretrained
        elif selected_layer == 'pretrained':
            layer_div_style = {'display': 'none'}

        return layer_div_style

    @app.callback(
        Output("tabs", "value"),
        [Input("train-button", "n_clicks")],
        [State("tabs", "value")]
    )
    # Callback function executed when the train button is clicked
    def switch_tab(n_clicks, current_tab):
        # Switch to the results tab if the train button is clicked
        if n_clicks is None:
            return current_tab
        else:
            return "results-tab"

def get_layer_details(layer_type, index):
    if layer_type == "Conv2D":
        # Create a Div element for Conv2D layer details
        return html.Div(
            style={"display": "flex", "align-items": "center"},
            children=[
                # Input field for number of filters
                html.Label("Num Filters:", style={"margin": "0px 5px 0px 15px"}),
                dcc.Input(
                    value=32 if index == 0 else 64, # Default values for pre-defined layers based on index
                    type="number",
                    id={"type": "num-filters", "index": index},
                    style={"margin": "0px 5px 0px 5px", "width": "100px"}
                ),
                # Input field for kernel shape
                html.Label("Kernel Shape:", style={"margin": "0px 5px 0px 5px"}),
                dcc.Input(
                    value="(3, 3)",
                    type="text",
                    id={"type": "kernel-shape", "index": index},
                    style={"margin": "0px 5px 0px 5px", "width": "100px"}
                ),
                # Dropdown for activation function
                html.Label("Activation Function:", style={"margin": "0px 0px 0px 5px"}),
                dcc.Dropdown(
                    options=[
                        {"label": "ReLU", "value": "relu"},
                        {"label": "Sigmoid", "value": "sigmoid"},
                        {"label": "Softmax", "value": "softmax"}
                    ],
                    value="relu",
                    clearable=False,
                    id={"type": "activation", "index": index},
                    style={"width": "100px", "margin": "0px 0px 0px 5px"}
                )
            ]
        )
    elif layer_type == "Dropout":
        # Create a Div element for Dropout layer details
        return html.Div(
            style={"display": "flex", "align-items": "center"},
            children=[
                # Input field for dropout rate
                html.Label("Dropout Rate:", style={"margin": "0px 5px 0px 15px"}),
                dcc.Input(
                    value=0.25,
                    type="number",
                    id={"type": "dropout-rate", "index": index},
                    style={"margin": "0px -5px 0px 5px", "width": "100px"}
                )
            ]
        )
    elif layer_type == "MaxPooling2D":
        # Create a Div element for MaxPooling2D layer details
        return html.Div(
            style={"display": "flex", "align-items": "center"},
            children=[
                # Input field for pool size
                html.Label("Pool Size:", style={"margin": "0px 5px 0px 15px"}),
                dcc.Input(
                    value="(2, 2)",
                    type="text",
                    id={"type": "pool-size", "index": index},
                    style={"margin": "0px -5px 0px 5px", "width": "100px"}
                )
            ]
        )
    elif layer_type == "Flatten":
        # Return None for Flatten layer, as no additional details are required
        return None
    elif layer_type == "Dense":
        # Create a Div element for Dense layer details
        return html.Div(
            style={"display": "flex", "align-items": "center"},
            children=[
                # Input field for number of units
                html.Label("Num Units:", style={"margin": "0px 5px 0px 15px"}),
                dcc.Input(
                    value=10 if index == 5 else 128,
                    type="number",
                    id={"type": "num-units", "index": index},
                    style={"margin": "0px 5px 0px 5px", "width": "100px"}
                ),
                # Dropdown for activation function
                html.Label("Activation Function:", style={"margin": "0px 0px 0px 5px"}),
                dcc.Dropdown(
                    options=[
                        {"label": "ReLU", "value": "relu"},
                        {"label": "Sigmoid", "value": "sigmoid"},
                        {"label": "Softmax", "value": "softmax"}
                    ],
                    value="softmax" if index == 5 else "relu",
                    clearable=False,
                    id={"type": "activation", "index": index},
                    style={"width": "100px", "margin": "0px 0px 0px 5px"}
                )
            ]
        )
    else:
        # Return an empty Div element if layer_type is not defined
        return html.Div()

def create_layer_div(index):
    # Create a Div element for each custom layer
    layer = html.Div(
        id={"type": "layer", "index": index},
        style={"display": "flex", "align-items": "center", "margin": "10px 10px 0px 5px"},
        children=[
            # Dropdown for selecting the layer type
            dcc.Dropdown(
                options=layer_options,
                # Set the default layer type based on the index
                value="Conv2D" if index == 0 else "Dropout" if index == 1 else "MaxPooling2D" if index == 2 else "Flatten" if index == 3 else "Dense",
                id={"type": "layer-type", "index": index},
                style={"width": "160px", "margin": "0px 0px 0px 5px"}
            ),
            # Div element containing layer details based on selected layer type
            html.Div(
                id={"type": "layer-details", "index": index},
                # Get the layer details based on the selected layer type and index
                children=get_layer_details("Conv2D" if index == 0 else "Dropout" if index == 1 else "MaxPooling2D" if index == 2 else "Flatten" if index == 3 else "Dense", index)
            ),
            # Button to move layer up
            html.Button('\u2191', className="up", id={"type": "up-button", "index": index}, style={"margin": "0px 5px 0px 15px"}),
            # Button to move layer down
            html.Button('\u2193', className="down", id={"type": "down-button", "index": index}, style={"margin": "0px 5px 0px 5px"}),
            # Button to delete layer
            html.Button("Delete", id={"type": "delete-button", "index": index}, n_clicks=0, style={"margin": "0px 5px 0px 5px"})
        ]
    )
    return layer

def extract_layer_params(layer_details):
    params = []

    if isinstance(layer_details, list):
        for child in layer_details:
            # Recursively call extract_layer_params for each element in the list
            params.append(extract_layer_params(child))

    elif isinstance(layer_details, dict):
        for key, value in layer_details["props"].items():
            if key == "children":
                # Recursively call extract_layer_params for the nested element if the key is "children"
                params.append(extract_layer_params(value))
            elif key == "value":
                if isinstance(value, str):
                    if value.startswith("'") and value.endswith("'"):
                        # Remove single quotes if present at the beginning and end of the string value
                        value = value[1:-1]
                    elif value == "relu" or value == "softmax" or value == "sigmoid":
                        # For activation functions (relu, sigmoid, softmax), wrap the value with 'activation='
                        value = "activation=" + f"'{value}'"
                if value:
                    # If the value is not empty after processing, add it to the params list
                    params.append(value)

    elif isinstance(layer_details, str):
        if ":" in layer_details:
            # Extract the part after the colon and remove leading/trailing spaces
            layer_details = layer_details.split(":")[1].strip()

            if layer_details.startswith("'") and layer_details.endswith("'"):
                # Remove single quotes at the beginning and end of the string value
                layer_details = layer_details[1:-1]

        if layer_details:
            # Add layer_details to params list
            params.append(layer_details)

    return ", ".join(str(param) for param in params if param)

def generate_layer_code(layer_type, layer_details):
    param_str = extract_layer_params(layer_details)

    # Generate the code for the layer based on the layer type
    if layer_type == "Conv2D":
        return f"Conv2D({param_str})"
    elif layer_type == "Dropout":
        return f"Dropout({param_str})"
    elif layer_type == "MaxPooling2D":
        return f"MaxPooling2D({param_str})"
    elif layer_type == "Flatten":
        return "Flatten()"
    elif layer_type == "Dense":
        return f"Dense({param_str})"
    else:
        return ""

def create_keras_model(layers):
    # Create a Sequential model and add the layers to it
    model = Sequential()
    for layer_config in layers:
        layer_instance = eval(layer_config)
        model.add(layer_instance)
    return model

def create_pretrained_model():
    # Load the VGG16 model trained on ImageNet
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

    # Get the output of the last convolutional layer
    last_conv_layer = base_model.get_layer('block5_conv3')

    # Create a new model that outputs the feature maps of the last convolutional layer
    model = Model(inputs=base_model.input, outputs=last_conv_layer.output)

    # Freeze the weights of the base model
    model.trainable = False

    # Add layers for classification of MNIST in addition to the base model
    x = tf.keras.layers.GlobalAveragePooling2D()(last_conv_layer.output)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

    # Create the final model
    final_model = Model(inputs=model.input, outputs=outputs)

    return final_model

def load_data(data_augmentation):
    # Split the data into train and test sets (80% train, 20% test)
    train_images, test_images, train_labels, test_labels = train_test_split(
        VIAL.images_outliers_removed, VIAL.labels_outliers_removed, test_size=0.2, random_state=42)

    # Split the test set into test and validation sets (50% test, 50% validation)
    test_images, val_images, test_labels, val_labels = train_test_split(
        test_images, test_labels, test_size=0.5, random_state=42)

    train_images_dataset = tf.data.Dataset.from_tensor_slices(train_images)

    if data_augmentation is not None:
        # Apply data augmentation to the train dataset
        train_images_dataset = train_images_dataset.map(lambda x: augment_data(x, data_augmentation))

    train_images = np.array(list(train_images_dataset.as_numpy_iterator()))

    train_images = np.array(train_images)
    train_labels = np.array(train_labels)

    val_images = np.array(val_images)
    val_labels = np.array(val_labels)

    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    return train_images, train_labels, test_images, test_labels, val_images, val_labels

def load_test_data():
    # Load MNIST dataset from TensorFlow Datasets
    dataset = tfds.load('mnist', split='test', data_dir=r'C:/mnist')

    images = []
    labels = []

    # Iterate over the dataset and collect images
    for img_label in dataset:
        image = img_label['image']
        images.append(image.numpy())
        labels.append(img_label['label'].numpy())

    images = np.array(images)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels

def augment_data(image, data_augmentation):
    if "random-left-right" in data_augmentation:
        # Flip the image from left to right
        image = tf.image.random_flip_left_right(image)
    if "random-up-down" in data_augmentation:
        # Flip the image from up to down
        image = tf.image.random_flip_up_down(image)
    if "random-rotation" in data_augmentation:
        # Rotate the image randomly by 0, 90, 180, or 270 degrees
        image = tf.image.rot90(image, k=random.randint(0, 3))
    return image

def preprocess_data(image, label):
    # Cast the image to uint8
    image = tf.cast(image, tf.uint8)
    return image, label

def maml(final_model, num_inner_updates, learning_rate, batch_size, num_episodes, data_augmentation, pretrained_model):
    # Load and preprocess images and labels
    train_images, train_labels, test_images, test_labels, val_images, val_labels = load_data(data_augmentation)
    ground_truth_test_images, ground_truth_test_labels = load_test_data()

    # Resize the images to 32x32 if using a pretrained model
    if pretrained_model:
        train_images = tf.image.resize(train_images, (32, 32))
        test_images = tf.image.resize(test_images, (32, 32))
        ground_truth_test_images = tf.image.resize(ground_truth_test_images, (32, 32))
        val_images = tf.image.resize(val_images, (32, 32))

    train_images, train_labels = preprocess_data(train_images, train_labels)
    test_images, test_labels = preprocess_data(test_images, test_labels)
    ground_truth_test_images, ground_truth_test_labels = preprocess_data(val_images, val_labels)
    val_images, val_labels = preprocess_data(val_images, val_labels)

    train_images = tf.image.grayscale_to_rgb(images=tf.convert_to_tensor(train_images))
    test_images = tf.image.grayscale_to_rgb(images=tf.convert_to_tensor(test_images))
    ground_truth_test_images = tf.image.grayscale_to_rgb(images=tf.convert_to_tensor(ground_truth_test_images))
    val_images = tf.image.grayscale_to_rgb(images=tf.convert_to_tensor(val_images))

    # Safe test images and labels in global variables for XAI
    global test_images_exp
    test_images_exp = test_images

    global test_labels_exp
    test_labels_exp = test_labels

    # Convert the numpy arrays to TensorFlow Dataset objects
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))

    # Randomize order of training data
    train_dataset = train_dataset.shuffle(buffer_size=10000).batch(batch_size)

    # Batch the validation dataset into mini-batches
    val_dataset = val_dataset.batch(batch_size)

    # Use ADAM optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Define a separate optimizer for the inner loop updates
    inner_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Adapted from the solutions given in:
    # https://colab.research.google.com/github/mari-linhares/tensorflow-maml/blob/master/maml.ipynb#scrollTo=xzVi0_YfB2aZ
    # https://www.kaggle.com/code/giangpt/maml-for-omniglot-with-dataloader
    # https://www.kaggle.com/code/prachi13/metagenomics-disease-using-few-shot-learning

    # Inner loop of MAML
    def inner_train_step(model, images, labels):
        with tf.GradientTape() as tape:
            logits = model(images, training=True)
            loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits)

        gradients = tape.gradient(loss, model.trainable_variables)
        inner_optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        return loss

    # Outer loop of MAML
    def outer_train_step(model, support_images, support_labels, query_images, query_labels):
        for _ in range(num_inner_updates):
            support_loss = inner_train_step(model, support_images, support_labels)

        # Evaluation on the query set
        with tf.GradientTape() as tape:
            query_logits = model(query_images, training=False)
            query_loss = tf.keras.losses.sparse_categorical_crossentropy(query_labels, query_logits)

        gradients = tape.gradient(query_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        support_loss = tf.reduce_mean(support_loss)
        query_loss = tf.reduce_mean(query_loss)

        return support_loss, query_loss

    def compute_accuracy(labels, logits):
        labels = tf.cast(labels, tf.int64)
        logits = tf.cast(logits, tf.int64)
        predictions = tf.argmax(logits, axis=1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, predictions), tf.float32))
        return accuracy

    # Create iterators for support and query datasets
    support_iterator = iter(train_dataset)
    query_iterator = iter(val_dataset)

    # Initialize empty list to accumulate episode HTML elements
    episode_html_list = []

    for episode in range(num_episodes):
        # Update support and query iterators at the beginning of each episode
        support_images, support_labels = next(support_iterator)
        query_images, query_labels = next(query_iterator)

        # Inner loop for updates on the support set
        for _ in range(num_inner_updates):
            inner_train_step(final_model, support_images, support_labels)

        # Outer loop (evaluation) using the updated model
        support_loss, query_loss = outer_train_step(final_model, support_images, support_labels, query_images,
                                                    query_labels)

        # Compute accuracy on the support and query sets using the final model
        support_accuracy = compute_accuracy(support_labels, final_model(support_images, training=False))
        query_accuracy = compute_accuracy(query_labels, final_model(query_images, training=False))

        # Store the results for the current episode
        episode_results = {
            'episode': episode + 1,
            'support_loss': tf.reduce_mean(support_loss).numpy(),
            'support_accuracy': support_accuracy.numpy(),
            'query_loss': tf.reduce_mean(query_loss).numpy(),
            'query_accuracy': query_accuracy.numpy(),
            'support_images': [VIAL.get_encoded_image(image) for image in support_images],
            'query_images': [VIAL.get_encoded_image(image) for image in query_images]
        }

        # Display the training results of the episode
        episode_html = [
            html.H3(f"Episode {episode_results['episode']}", style={"margin": "5px 15px 0px 15px"}),
            html.Div(f"Support Loss: {episode_results['support_loss']:.4f}", style={"margin": "5px 0px 0px 15px"}),
            html.Div(f"Support Accuracy: {episode_results['support_accuracy']:.4f}", style={"margin": "5px 0px 0px 15px"}),
            html.Div(f"Query Loss: {episode_results['query_loss']:.4f}", style={"margin": "5px 0px 0px 15px"}),
            html.Div(f"Query Accuracy: {episode_results['query_accuracy']:.4f}", style={"margin": "5px 0px 0px 15px"}),
            html.H5("Support Images", style={"margin": "5px 0px 0px 15px"}),
            html.Div([html.Img(src=f"data:image/png;base64,{img}", style={"margin": "5px"}) for img in episode_results['support_images']], style={"margin": "0px 0px 0px 10px"}),
            html.H5("Query Images", style={"margin": "5px 0px 0px 15px"}),
            html.Div([html.Img(src=f"data:image/png;base64,{img}", style={"margin": "5px"}) for img in episode_results['query_images']], style={"margin": "0px 0px 0px 10px"}),
            html.Hr()
        ]

        episode_html_list.append(html.Div(episode_html))

    # Calculate accuracy on labeled test data
    test_accuracy = Accuracy()
    batch_size = 32  # Batch size for accuracy calculation
    num_test_samples = len(test_labels)
    for i in range(0, num_test_samples, batch_size):
        images_batch = test_images[i:i + batch_size]
        labels_batch = test_labels[i:i + batch_size]

        logits = final_model(images_batch, training=False)
        predictions = tf.argmax(logits, axis=1)
        test_accuracy.update_state(labels_batch, predictions)

    labeled_test_accuracy = test_accuracy.result().numpy()

    # Calculate accuracy on ground truth test data
    test_accuracy = Accuracy()
    batch_size = 32  # Batch size for accuracy calculation
    num_test_samples = len(ground_truth_test_labels)
    for i in range(0, num_test_samples, batch_size):
        images_batch = ground_truth_test_images[i:i + batch_size]
        labels_batch = ground_truth_test_labels[i:i + batch_size]

        logits = final_model(images_batch, training=False)
        predictions = tf.argmax(logits, axis=1)
        test_accuracy.update_state(labels_batch, predictions)

    ground_truth_test_accuracy = test_accuracy.result().numpy()

    # Generate HTML element for test accuracy
    final_test_accuracy_html = html.Div([
        html.Div(f"Labeled Test Accuracy: {labeled_test_accuracy:.4f}", style={"margin": "0px 0px 15px 15px"}),
        html.Div(f"Ground Truth Test Accuracy: {ground_truth_test_accuracy:.4f}", style={"margin": "0px 0px 15px 15px"})
    ])

    return [
        dcc.Tab(label="Train Model", value="train-tab"),
        dcc.Tab(label="Training Results", value="results-tab"),
        dcc.Tab(label="Final Test Accuracy", value="accuracy-tab"),
        *episode_html_list,
        final_test_accuracy_html
    ]

def regular_model(final_model, batch_size, epochs, learning_rate, data_augmentation, pretrained_model):
    # Load and preprocess images and labels
    train_images, train_labels, test_images, test_labels, val_images, val_labels = load_data(data_augmentation)
    ground_truth_test_images, ground_truth_test_labels = load_test_data()

    # Resize images to 32x32 if using pretrained model VGG16
    if pretrained_model:
        train_images = tf.image.resize(train_images, (32, 32))
        test_images = tf.image.resize(test_images, (32, 32))
        ground_truth_test_images = tf.image.resize(ground_truth_test_images, (32, 32))
        val_images = tf.image.resize(val_images, (32, 32))

    train_images, train_labels = preprocess_data(train_images, train_labels)
    test_images, test_labels = preprocess_data(test_images, test_labels)
    ground_truth_test_images, ground_truth_test_labels = preprocess_data(ground_truth_test_images, ground_truth_test_labels)
    val_images, val_labels = preprocess_data(val_images, val_labels)

    train_images = tf.image.grayscale_to_rgb(images=tf.convert_to_tensor(train_images))
    test_images = tf.image.grayscale_to_rgb(images=tf.convert_to_tensor(test_images))
    ground_truth_test_images = tf.image.grayscale_to_rgb(images=tf.convert_to_tensor(ground_truth_test_images))
    val_images = tf.image.grayscale_to_rgb(images=tf.convert_to_tensor(val_images))

    # Safe test images and labels in global variables for XAI
    global test_images_exp
    test_images_exp = test_images

    global test_labels_exp
    test_labels_exp = test_labels

    # Use ADAM optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Compile the model
    final_model.compile(optimizer=optimizer,
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                        metrics=['accuracy'])

    # Train the model
    history = final_model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs,
                              validation_data=(val_images, val_labels))

    # Store the entire history
    history_dict = history.history

    # Generate HTML elements for each epoch
    epoch_results_html = []
    for epoch in range(epochs):
        epoch_number = epoch + 1

        train_loss = history_dict['loss'][epoch]
        train_accuracy = history_dict['accuracy'][epoch]
        val_loss = history_dict['val_loss'][epoch]
        val_accuracy = history_dict['val_accuracy'][epoch]

        epoch_html = [
            html.H3(f"Epoch {epoch_number}", style={"margin": "5px 15px 0px 15px"}),
            html.Div(f"Train Loss: {train_loss:.4f}", style={"margin": "5px 0px 0px 15px"}),
            html.Div(f"Train Accuracy: {train_accuracy:.4f}", style={"margin": "5px 0px 0px 15px"}),
            html.Div(f"Val Loss: {val_loss:.4f}", style={"margin": "5px 0px 0px 15px"}),
            html.Div(f"Val Accuracy: {val_accuracy:.4f}", style={"margin": "5px 0px 0px 15px"}),
            html.Hr()
        ]
        epoch_results_html.append(html.Div(epoch_html))

    # Calculate labeled test loss and accuracy
    labeled_test_loss, labeled_test_accuracy = final_model.evaluate(test_images, test_labels)

    # Calculate ground truth test loss and accuracy
    ground_truth_test_loss, ground_truth_test_accuracy = final_model.evaluate(ground_truth_test_images, ground_truth_test_labels)

    # Generate HTML element for final test accuracy
    final_test_accuracy_html = html.Div([
        html.Div(f"Labeled Test Accuracy: {labeled_test_accuracy:.4f}", style={"margin": "0px 0px 15px 15px"}),
        html.Div(f"Ground Truth Test Accuracy: {ground_truth_test_accuracy:.4f}", style={"margin": "0px 0px 15px 15px"})
    ])

    return [
        dcc.Tab(label="Train Model", value="train-tab"),
        dcc.Tab(label="Training Results", value="results-tab"),
        dcc.Tab(label="Final Test Accuracy", value="accuracy-tab"),
        *epoch_results_html,
        final_test_accuracy_html
    ]