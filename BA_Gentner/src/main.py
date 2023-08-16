import dash as dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html

import FSL
import VIAL
import XAI
from FSL import create_layer_div

# Create a Dash app instance
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# Define the layout of the app
app.layout = html.Div(
    children=[
        # Define tabs for different sections of the app
        dcc.Tabs(
            id='tabs',
            value='remove-outlier-tab',
            children=[
                # Tab for 'Remove Outliers' section
                dcc.Tab(
                    label='Remove Outliers',
                    value='remove-outlier-tab',
                    children=[

                        html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '20px'}, children=[

                            # Left side: Scatter Plot and MNIST Clusters
                            html.Div(
                                style={'display': 'flex', 'flexDirection': 'column'},
                                children=[
                                    dcc.Loading(
                                        dcc.Graph(id='scatter-plot', figure=VIAL.fig),
                                        type='default',
                                    ),
                                ]
                            ),
                            # Right side: Tabs
                            html.Div(
                                style={'display': 'flex', 'flexDirection': 'column'},
                                children=[
                                    # Dropdown to select clusters
                                    html.Div(id='cluster-dropdown-container', style={'margin': '10px 10px 10px 0px'},
                                             children=[
                                                 dcc.Dropdown(
                                                     id='cluster-dropdown',
                                                     options=[
                                                         {'label': cluster_name, 'value': cluster_name}
                                                         for cluster_name in VIAL.cluster_names[1:]
                                                     ],
                                                     value=None,
                                                     placeholder="Select a cluster",
                                                     style={'width': '200px'},
                                                 )
                                             ]),
                                    # Hidden div that contains tabs for MSE and Remove Individual Outliers
                                    html.Div(style={'display': 'none'}, id='tabs-div', children=[
                                        dcc.Tabs(
                                            value='mse-tab',
                                            children=[
                                                dcc.Tab(
                                                    label='Reduce MSE',
                                                    value='mse-tab',
                                                    children=[

                                                        html.Div(
                                                            id='line-chart-container',
                                                            children=[
                                                                dcc.Graph(id='mse-vs-num-images'),
                                                                html.H6('Please click on the chart to reduce the MSE threshold', style={'text-align': 'center'})
                                                            ],
                                                        )
                                                    ]
                                                ),
                                                dcc.Tab(
                                                    id='outlier-tab',
                                                    label='Remove Individual Outliers',
                                                    value='outlier-tab',
                                                    children=[
                                                        html.Div(
                                                            className='six columns',
                                                            children=[
                                                                html.Div(
                                                                    id='image-frame',
                                                                    className='image-frame',
                                                                    style={
                                                                        'max-height': '400px',
                                                                        'overflow-y': 'scroll',
                                                                        'padding': '10px',
                                                                        'display': 'flex',
                                                                        'flex-wrap': 'wrap',
                                                                        'margin': '0px 0px 10px 0px'
                                                                    }
                                                                ),
                                                                html.Button('Delete Selected Images',
                                                                            id='delete-button', n_clicks=0,
                                                                            style={'margin': '10px 10px 10px 0px'}),
                                                            ]
                                                        ),
                                                    ]
                                                )
                                            ])
                                    ]
                                            )
                                ]
                            )
                        ]),
                    ]),
                # Tab for 'Assign Labels' section
                dcc.Tab(
                    label='Assign Labels',
                    value='labels-tab',
                    id='labels-tab',
                    children=[
                        html.Div(id="images-labels"),
                        html.Button('Save Labels', id='save-labels-button', n_clicks=0,
                                    style={'margin': '15px'}),
                        html.Div(id="safe-info")
                    ]),
                # Tab for 'Train Model' section
                dcc.Tab(label="Train Model", value="train-tab", children=[

                    html.Div(style={"margin": "10px 15px"}, children=[
                        dcc.Checklist(options=[{'label': 'Random Flip Images Left Right', 'value': 'random-left-right'},
                                               {'label': 'Random Flip Images Up Down', 'value': 'random-up-down'},
                                               {'label': 'Random Rotation of Images 0, 90, 180, or 270 Degrees',
                                                'value': 'random-rotation'}],
                                      inputStyle={'margin-right': '10px'},
                                      id='data-augmentation-checklist'),
                    ]),
                    html.Div(style={"margin": "15px 0px 0px", "display": "flex", "align-items": "center"}, children=[
                        html.Label("Choose a Model:", style={"width": "133px", "margin": "0px 15px"}),
                        dcc.Dropdown(
                            id='model-dropdown',
                            options=[
                                {'label': 'Model-Agnostic Meta-Learning', 'value': 'maml'},
                                {'label': 'Regular Model', 'value': 'regular'},
                            ],
                            value='maml',
                            style={'width': '55%'}
                        ),
                    ]),
                    html.Div(style={"margin": "15px 0px 0px", "display": "flex", "align-items": "center"}, children=[
                        html.Label("Define Layers:", style={"width": "133px", "margin": "0px 15px"}),
                        dcc.Dropdown(
                            id='layers-dropdown',
                            options=[
                                {'label': 'Custom Layers', 'value': 'custom'},
                                {'label': 'Transfer Learning with VGG16 initialized on ImageNet',
                                 'value': 'pretrained'},
                            ],
                            value='custom',
                            style={'width': '55%'}
                        ),
                    ]),
                    html.Div([
                        html.Div(id="layer-container", children=[create_layer_div(i) for i in range(6)]),
                        html.Div(id="total-layers", style={"display": "none"}),
                        html.Button("Add Layer", id="add-layer-button", n_clicks=0, style={"margin": "10px 15px -5px"}),
                    ], id='layer-div'),

                    html.Div(style={"margin": "0px 15px"}, children=[
                        html.Label("Number of Episodes:", style={"width": "200px", "display": "inline-block"}),
                        dcc.Input(id="num-episodes-input", type="number", value=20, style={"width": "100px"}),
                    ], id='num-episodes-div'),
                    html.Div(style={"margin": "15px 15px"}, children=[
                        html.Label("Number of Inner Updates:", style={"width": "200px", "display": "inline-block"}),
                        dcc.Input(id="num-inner-updates-input", type="number", value=16, style={"width": "100px"}),
                    ], id='num-inner-updates-div'),
                    html.Div(style={"margin": "15px 15px"}, children=[
                        html.Label("Number of Epochs:", style={"width": "200px", "display": "inline-block"}),
                        dcc.Input(id="num-epochs-input", type="number", value=1, style={"width": "100px"}),
                    ], id='num-epochs-div'),
                    html.Div(style={"margin": "15px 15px"}, children=[
                        html.Label("Learning Rate:", style={"width": "200px", "display": "inline-block"}),
                        dcc.Input(id="learning-rate-input", type="number", step="any", value=0.001,
                                  style={"width": "100px"}),
                    ]),
                    html.Div(style={"margin": "15px 15px 15px"}, children=[
                        html.Label("Batch Size:", style={"width": "200px", "display": "inline-block"}),
                        dcc.Input(id="batch-size-input", type="number", value=256, style={"width": "100px"}),
                    ]),

                    html.Button("Train Model", id="train-button", style={"margin": "0px 15px 10px"}),
                ]),
                # Tab for 'Training Results' section
                dcc.Tab(label="Training Results", value="results-tab", children=[
                    html.Div(
                        id="alert-output",
                        style={"display": "flex", "justify-content": "center", "align-items": "center",
                               "position": "relative",
                               "height": "100vh"},
                        children=[
                            dcc.Loading(
                                id="loading-training-results",
                                type="default",
                                fullscreen=True,
                                style={"position": "absolute",
                                       "top": "50%",
                                       "left": "50%",
                                       "transform": "translate(-50%, -50%)"},
                                children=[
                                    html.Div(
                                        children=[
                                            html.Div(id="training-results"),
                                            html.Div(id="alert-no-training"),
                                            html.Div(id="alert-output")
                                        ], style={"position": "absolute",
                                                  "top": "5px",
                                                  "left": "0",
                                                  "right": "0",
                                                  }
                                    )
                                ],
                            ),
                        ],
                    )
                ]),
                # Tab for 'Explain Model' section
                dcc.Tab(label="Explain Model", value="explain-tab", id="explain-tab", children=[

                    html.Div([
                        html.Div(id="alert-no-explanation", style={"margin-top": "5px"}),
                        html.Div(id="cm-div"),

                        dcc.Loading(
                            id="loading-explanation-results",
                            type="default",
                            fullscreen=False,
                            style={"position": "relative"},
                            children=[
                                html.Div(id='image-container', style={"height": "50vh"})
                            ],
                        )
                    ])
                ])
            ])
    ])

# Register callbacks for the app from custom modules
VIAL.register_callbacks_vial(app)
FSL.register_callbacks_fsl(app)
XAI.register_callbacks_xai(app)

# Run the app
if __name__ == '__main__':
    app.run_server(debug=False)