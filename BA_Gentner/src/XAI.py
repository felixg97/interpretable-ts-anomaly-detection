import base64
import cv2
from io import BytesIO

from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

from lime import lime_image
from skimage.segmentation import mark_boundaries
from lime.wrappers.scikit_image import SegmentationAlgorithm
import shap
import shap.plots

# Initialize global variables
global final_model
final_model = None

global test_images
test_images = None

global test_labels
test_labels = None

global cm
cm = None

global predicted_labels
predicted_labels = None

def register_callbacks_xai(app):
    @app.callback(
        Output('image-container', 'children'),
        [Input('confusion-matrix', 'clickData')]
    )
    # Callback executed when cell in confusion matrix is clicked
    def display_images(click_data):
        if click_data is not None and final_model is not None:
            # Get the true and predicted class from the click data on the confusion matrix
            true_class = click_data['points'][0]['y']
            pred_class = click_data['points'][0]['x']

            # Initialize the predicted data
            pred_class_label = ": " + str(pred_class)
            pred_class_org = pred_class
            second_text = ""

            # Select up to 5 images from the selected cell in the confusion matrix
            indices = np.where((test_labels == true_class) & (predicted_labels == pred_class))[0]
            images = np.array(test_images)[indices[:5]]

            # Initialize images to display
            images_html = []

            images_grad_cam_pred = []
            images_grad_cam_true = []

            images_lime_pred = []
            images_lime_true = []

            images_shap = []

            # Initialize the LIME explainer
            lime_explainer = lime_image.LimeImageExplainer(random_state=5)

            # Initialize the SHAP explainer
            masker = shap.maskers.Image(mask_value="blur(128, 128)", shape=test_images[0].shape)
            shap_explainer = shap.Explainer(model=final_model.predict, masker=masker, algorithm="auto",
                                            output_names=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

            for image in images:
                # Predict the second most likely class if the true class is the same as the predicted class
                if true_class == pred_class_org:
                    pred_image = final_model.predict(np.expand_dims(image, axis=0))
                    sorted_predictions = np.argsort(-pred_image)
                    pred_class = sorted_predictions[0][1]
                    pred_class_label = ""
                    second_text = "2nd "

                # Create HTML element of the original image
                rescaled_image = image
                buffer = BytesIO()
                plt.imshow(rescaled_image)
                plt.axis('off')
                plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight', pad_inches=0)
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                image_html = html.Img(
                    src='data:image/png;base64,' + image_base64,
                    style={'width': '150px', 'margin': '5px'}
                )
                images_html.append(image_html)

                # Grad-CAM
                first_image = np.expand_dims(image, axis=0)

                # Get Grad-CAM heatmap for the true class
                icam_true = GradCAM(final_model, true_class, find_target_layer(final_model))
                heatmap_true = icam_true.compute_heatmap(first_image)

                # Get Grad-CAM heatmap for the predicted class
                icam_pred = GradCAM(final_model, pred_class, find_target_layer(final_model))
                heatmap_pred = icam_pred.compute_heatmap(first_image)

                # Create HTML element of the true Grad-CAM heatmap
                buffer = BytesIO()
                plt.imshow(heatmap_true, alpha=0.8, extent=(0, image.shape[1], image.shape[0], 0))
                plt.axis('off')
                plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight', pad_inches=0)
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                image_html = html.Img(
                    src='data:image/png;base64,' + image_base64,
                    style={'width': '150px', 'margin': '5px'}
                )
                images_grad_cam_true.append(image_html)

                # Create HTML element of the predicted Grad-CAM heatmap
                buffer = BytesIO()
                plt.imshow(heatmap_pred, alpha=0.4, extent=(0, image.shape[1], image.shape[0], 0))

                plt.axis('off')
                plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight', pad_inches=0)
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                image_html = html.Img(
                    src='data:image/png;base64,' + image_base64,
                    style={'width': '150px', 'margin': '5px'}
                )
                images_grad_cam_pred.append(image_html)

                # LIME
                # Create a segmentation algorithm
                segmenter = SegmentationAlgorithm(algo_type="quickshift",
                                                  kernel_size=1,
                                                  max_dist=2)

                # Create the LIME explanation
                lime_explanation = lime_explainer.explain_instance(image=image, classifier_fn=final_model.predict,
                                                                   top_labels=10, num_samples=500,
                                                                   segmentation_fn=segmenter, random_seed=5)

                # Get LIME image and heatmap for the true class
                image_true, mask_true = lime_explanation.get_image_and_mask(label=true_class,
                                                                           positive_only=False,
                                                                           negative_only=False,
                                                                           hide_rest=False)

                # Get LIME image and heatmap for the predicted class
                image_pred, mask_pred = lime_explanation.get_image_and_mask(label=pred_class,
                                                                           positive_only=False,
                                                                           negative_only=False,
                                                                           hide_rest=False)

                # Create HTML element of the predicted LIME heatmap
                rescaled_image = (mark_boundaries(image=image_pred / 2 + 0.5, label_img=mask_pred)).clip(0, 1)
                buffer = BytesIO()
                plt.imshow(rescaled_image)
                plt.axis('off')
                plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight', pad_inches=0)
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                image_html = html.Img(
                    src='data:image/png;base64,' + image_base64,
                    style={'width': '150px', 'margin': '5px'}
                )
                images_lime_pred.append(image_html)

                # Create HTML element of the true LIME heatmap
                rescaled_image = (mark_boundaries(image=image_true / 2 + 0.5, label_img=mask_true)).clip(0, 1)
                buffer = BytesIO()
                plt.imshow(rescaled_image)
                plt.axis('off')
                plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight', pad_inches=0)
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                image_html = html.Img(
                    src='data:image/png;base64,' + image_base64,
                    style={'width': '150px', 'margin': '5px'}
                )
                images_lime_true.append(image_html)

                # SHAP
                shap_values = shap_explainer(first_image, max_evals=500, outputs=[true_class, pred_class])

                # Generate the SHAP image plot and create the image HTML element
                buffer = BytesIO()
                plt.clf()
                shap.image_plot(shap_values=shap_values, show=False)
                plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight', pad_inches=0)
                buffer.seek(0)
                plt.clf()
                image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                image_html = html.Img(
                    src='data:image/png;base64,' + image_base64,
                    style={'width': '306px', 'margin': '5px'}
                )
                images_shap.append(image_html)
                buffer.close()

            if images_html:
                # Create a Dash HTML Div with organized image displays for original, Grad-CAM, LIME, and SHAP images
                return html.Div(style={'display': 'flex', 'justify-content': 'center'}, children=[
                    html.Div(style={'display': 'grid', 'grid-template-columns': 'repeat(6, auto)', 'gap': '20px'},
                             children=[
                                 html.Div(
                                     style={'display': 'flex', 'flex-direction': 'column', 'align-items': 'center'},
                                     children=[
                                         html.H6('Original Images',
                                                 style={'display': 'flex', 'justify-content': 'center',
                                                        'align-items': 'center', 'flex-direction': 'column',
                                                        'height': '46px'}),
                                         html.Div(images_html,
                                                  style={'display': 'flex', 'flex-direction': 'column',
                                                         'align-items': 'center'})
                                     ]),
                                 html.Div(
                                     style={'display': 'flex', 'flex-direction': 'column', 'align-items': 'center'},
                                     children=[
                                         html.H6('Grad-CAM'),
                                         html.H6(f'Labeled Class: {true_class}'),
                                         html.Div(images_grad_cam_true,
                                                  style={'display': 'flex', 'flex-direction': 'column',
                                                         'align-items': 'center'})
                                     ]),
                                 html.Div(
                                     style={'display': 'flex', 'flex-direction': 'column', 'align-items': 'center'},
                                     children=[
                                         html.H6('Grad-CAM'),
                                         html.H6(f'{second_text}Predicted Class{pred_class_label}'),
                                         html.Div(images_grad_cam_pred,
                                                  style={'display': 'flex', 'flex-direction': 'column',
                                                         'align-items': 'center'})
                                     ]),
                                 html.Div(
                                     style={'display': 'flex', 'flex-direction': 'column', 'align-items': 'center'},
                                     children=[
                                         html.H6('LIME'),
                                         html.H6(f'Labeled Class: {true_class}'),
                                         html.Div(images_lime_true,
                                                  style={'display': 'flex', 'flex-direction': 'column',
                                                         'align-items': 'center'}),
                                         html.Div(
                                             style={'display': 'flex', 'flex-direction': 'column',
                                                    'align-items': 'center'},
                                             children=[
                                                 html.Div(
                                                     style={'background-color': '#8dfc83', 'width': '50px',
                                                            'height': '20px', 'margin': '5px'},
                                                     children=[]),
                                                 html.P('Green Area: Positive Impact',style={'font-size': '11px'})
                                             ]
                                         ),
                                         html.Div(
                                             style={'display': 'flex', 'flex-direction': 'column',
                                                    'align-items': 'center'},
                                             children=[
                                                 html.Div(
                                                     style={'background-color': '#fbfb35', 'width': '50px',
                                                            'height': '20px', 'margin': '5px'},
                                                     children=[]),
                                                 html.P('Yellow Area: Neutral Impact',style={'font-size': '11px'})
                                             ]
                                         ),
                                         html.Div(
                                             style={'display': 'flex', 'flex-direction': 'column',
                                                    'align-items': 'center'},
                                             children=[
                                                 html.Div(
                                                     style={'background-color': '#f68382', 'width': '50px',
                                                            'height': '20px', 'margin': '5px'},
                                                     children=[]),
                                                 html.P('Red Area: Negative Impact',style={'font-size': '11px'})
                                             ]
                                         ),
                                     ]),
                                 html.Div(
                                     style={'display': 'flex', 'flex-direction': 'column', 'align-items': 'center'},
                                     children=[
                                         html.H6('LIME'),
                                         html.H6(f'{second_text}Predicted Class{pred_class_label}'),
                                         html.Div(images_lime_pred,
                                                  style={'display': 'flex', 'flex-direction': 'column',
                                                         'align-items': 'center'})
                                     ]),

                                 html.Div(
                                     style={'display': 'flex', 'flex-direction': 'column', 'align-items': 'center'},
                                     children=[
                                         html.H6('SHAP',
                                                 style={'display': 'flex', 'justify-content': 'center',
                                                        'align-items': 'center', 'flex-direction': 'column',
                                                        'height': '46px'}),
                                         html.Div(images_shap,
                                                  style={'display': 'flex', 'flex-direction': 'column',
                                                         'align-items': 'center'})
                                     ])
                             ])
                ])

        # Return an empty Div element if confusion matrix is not clicked
        return html.Div()

def load_cm(model, images, labels):
    global final_model, test_images, test_labels, cm, predicted_labels
    final_model = model
    test_images = images
    test_labels = labels

    # Predict the test images
    predictions = final_model.predict(test_images)
    predicted_labels = np.argmax(predictions, axis=1)

    # Create a confusion matrix
    cm = confusion_matrix(test_labels, predicted_labels)

    # Create a Dash Graph to display the confusion matrix
    return html.Div([
        dcc.Graph(
        id='confusion-matrix',
        figure={
            'data': [
                go.Heatmap(
                    z=cm,
                    x=list(range(10)),
                    y=list(range(10)),
                    colorscale='Blues',
                    hoverinfo='text',
                    text=cm.astype(str),
                    showscale=False,
                    colorbar=dict(tickvals=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], ticktext=list(range(10))),
                    zmin=0,
                    zmax=np.max(cm),
                    hovertemplate='True Label: %{y}<br>Predicted Label: %{x}<br>Count: %{text}<extra></extra>'
                )
            ],
            'layout': go.Layout(
                title='Confusion Matrix',
                xaxis={'title': 'Predicted Label', 'tickmode': 'array', 'tickvals': list(range(10)),
                       'ticktext': list(range(10))},
                yaxis={'title': 'True Label', 'tickmode': 'array', 'tickvals': list(range(10)),
                       'ticktext': list(range(10)), 'autorange': 'reversed'},
                annotations=[
                    go.layout.Annotation(
                        x=x_val,
                        y=y_val,
                        text=str(cm[y_val][x_val]),  # Fix the order of indices for accessing confusion matrix elements
                        showarrow=False,
                        font=dict(color='black')
                    )
                    for y_val in range(10)
                    for x_val in range(10)
                ]
            )
        }
    ),
    html.H6('Please select a cell in the confusion matrix to display explanations',
            style={'text-align': 'center', 'margin-bottom': '20px'})
    ])


# Adapted from the solution given in:
# https://stackoverflow.com/questions/66182884/how-to-implement-grad-cam-on-a-trained-network
class GradCAM:
    def __init__(self, model, classIdx, layerName=None):
        # Store the model, class index, and layer to be used when visualizing the class activation map
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName

    def compute_heatmap(self, image, eps=1e-8):
        # Define the gradient model by specifying the inputs, outputs of the final 4D layer, and outputs of the last layer
        gradModel = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output, self.model.output])

        with tf.GradientTape() as tape:
            # Cast the image tensor to float-32
            inputs = tf.cast(image, tf.float32)

            # Predict the image using the gradient model
            (convOutputs, predictions) = gradModel(inputs)
            loss = predictions[:, self.classIdx]

        # Compute the gradients using automatic differentiation
        grads = tape.gradient(loss, convOutputs)

        # Compute the guided gradients
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads

        # Discard the batch dimension
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]

        # Compute the weights by averaging the guided gradients over all spatial dimensions
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        # Resize the heatmap to the size of the input image
        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))

        # Normalize the heatmap to the range [0, 1]
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")

        # Apply a color map to the heatmap
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_VIRIDIS)

        return heatmap

def find_target_layer(model):
    # Loop over the layers in reverse order to find the final convolutional layer in the network
    for layer in reversed(model.layers):
        # Check if the layer has a 4D output
        if len(layer.output_shape) == 4:
            return layer.name
    raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")
