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


def create_evaluation_bar_chart_1(cnn_results, svm_results, rf_results, knn_results, title, x_labels, x_axis_label,
                                  y_axis_label):
    """Creates a bar chart showing how well the CNN, SVM, RF, and KNN models performed based on the number of
    locations images were used from in the dataset.

    Args:
        cnn_results (list[int]): The results that the CNN model achieved per location.
        svm_results (list[int]): The results that the SVM model achieved per location.
        rf_results (list[int]): The results that the RF model achieved per location.
        knn_results (list[int]): The results that the KNN model achieved per location.
        title (str): The title of the bar chart.
        x_labels (list[str]): Five labels that will be shown along the x-axis of the bar chart.
        x_axis_label (str): The label displayed on the y-axis of the bar chart.
        y_axis_label (str): The label displayed on the y-axis of the bar chart.

    """
    red = '#ff757d'
    blue = '#4cb5f5'
    orange = '#ffb81f'
    purple = '#d472cd'

    fig = go.Figure(layout_yaxis_range=[0, 108])
    fig.add_trace(go.Bar(
        x=x_labels,
        y=cnn_results,
        name='CNN',
        marker_color=red,
        text=cnn_results,
        textposition='outside',
        texttemplate='%{text:.3}'
    ))
    fig.add_trace(go.Bar(
        x=x_labels,
        y=svm_results,
        name='SVM',
        marker_color=blue,
        text=svm_results,
        textposition='outside',
        texttemplate='%{text:.3}'
    ))
    fig.add_trace(go.Bar(
        x=x_labels,
        y=rf_results,
        name='RF',
        marker_color=orange,
        text=rf_results,
        textposition='outside',
        texttemplate='%{text:.3}'
    ))
    fig.add_trace(go.Bar(
        x=x_labels,
        y=knn_results,
        name='KNN',
        marker_color=purple,
        text=knn_results,
        textposition='outside',
        texttemplate='%{text:.3}'
    ))

    fig.update_xaxes(title_text=x_axis_label)
    fig.update_yaxes(title_text=y_axis_label)
    fig.update_layout(barmode='group',
                      title={
                          'text': title,
                          'x': 0.5,
                          'xanchor': 'center',
                          'yanchor': 'top'})
    fig.show()


def create_evaluation_bar_chart_2(results, title, y_axis_label):
    """Creates a bar chart.

    Args:
        results (list[int]): The results that the models achieved.
        title (str): The title of the bar chart.
        y_axis_label (str): The label displayed on the y-axis of the bar chart.

    """
    red = '#ff757d'
    blue = '#4cb5f5'
    orange = '#ffb81f'
    purple = '#d472cd'

    fig = go.Figure(layout_yaxis_range=[0, 108])
    fig.add_trace(go.Bar(
        x=['CNN', 'SVM', 'RF', 'KNN'],
        y=results,
        marker_color=[red, blue, orange, purple],
        text=results,
        textposition='outside',
        texttemplate='%{text:.3}'
    ))

    fig.update_xaxes(title_text='Models')
    fig.update_yaxes(title_text=y_axis_label)
    fig.update_layout(title={
        'text': title,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
    fig.show()
