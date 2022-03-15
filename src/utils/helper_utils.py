"""This module contains utility functions that help with various areas of development."""

import plotly.graph_objects as go

def get_channels(color_mode):
    """Returns the number of channels that a color mode has.

    Args:
        color_mode (str): The color mode. Either 'grayscale', 'rgb', or 'rgba'.

    Returns:
        int: The number of channels.
    """
    if color_mode == 'grayscale':
        return 1
    elif color_mode == 'rgb':
        return 3
    else:
        return 4


def get_seed(seeds, num_locations):
    """Returns the seed at a particular index in a list of seeds.

    E.g. if num_locations is 4, the 4th seed in the list will be returned.

    Args:
        seeds (list[int]): A list of seeds.
        num_locations (int): The number of locations that images are being loaded from.

    Returns:
        seed (int): A seed.

    """
    if num_locations <= 0:
        raise ValueError("The number of locations must be greater than 0.")
    if num_locations > len(seeds):
        raise ValueError(f"The number of locations is too high. Maximum = {len(seeds)}")

    return seeds[num_locations - 1]


def create_binary_bar_chart(cnn_results, svm_results, rf_results, knn_results):
    """Creates a bar chart showing how well the binary CNN, SVM, RF, and KNN models performed based on the number of
    locations images were used from.

    Args:
        cnn_results (list[int]): The results that the binary CNN model achieved per location.
        svm_results (list[int]): The results that the binary SVM model achieved per location.
        rf_results (list[int]): The results that the binary RF model achieved per location.
        knn_results (list[int]): The results that the binary KNN model achieved per location.

    """

    red = '#ff757d'
    blue = '#4cb5f5'
    orange = '#ffb81f'
    purple = '#d472cd'

    x = ['1', '2', '3', '4', '5']
    cnn_results = cnn_results
    svm_results = svm_results
    rf_results = rf_results
    knn_results = knn_results

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x,
        y=cnn_results,
        name='CNN',
        marker_color=red,
        text=cnn_results,
        textposition='outside',
        texttemplate='%{text:.3}'
    ))
    fig.add_trace(go.Bar(
        x=x,
        y=svm_results,
        name='SVM',
        marker_color=blue,
        text=svm_results,
        textposition='outside',
        texttemplate='%{text:.3}'
    ))
    fig.add_trace(go.Bar(
        x=x,
        y=rf_results,
        name='RF',
        marker_color=orange,
        text=rf_results,
        textposition='outside',
        texttemplate='%{text:.3}'
    ))
    fig.add_trace(go.Bar(
        x=x,
        y=knn_results,
        name='KNN',
        marker_color=purple,
        text=knn_results,
        textposition='outside',
        texttemplate='%{text:.3}'
    ))

    fig.update_xaxes(title_text="Number of locations")
    fig.update_yaxes(title_text="Accuracy (%)")
    fig.update_layout(barmode='group',
                      title={
                          'text': 'Accuracy of Binary Models on Magicseaweed Datasets (Best Seeds)',
                          'x': 0.5,
                          'xanchor': 'center',
                          'yanchor': 'top'})
    fig.show()