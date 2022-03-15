"""This module contains functions that create graphs to show the results that were gathered for the binary
classification models during the evaluation stage of the project. """

import plotly.graph_objects as go

red = '#ff757d'
blue = '#4cb5f5'
orange = '#ffb81f'
purple = '#d472cd'


def magicseaweed_binary_accuracy_chart():
    """Creates a bar chart showing...
    """

    x = ['1', '2', '3', '4', '5']
    cnn_results = [100, 99, 99, 99, 99]
    svm_results = [98, 96, 94, 95, 94]
    rf_results = [97, 95, 93, 94, 94]
    knn_results = [78, 74, 78, 76, 77]

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


def magicseaweed_binary_precision_chart():
    """Creates a bar chart showing...
    """
    print('Test')


def magicseaweed_binary_recall_chart():
    """Creates a bar chart showing...
    """
    print('Test')


def surfline_binary_accuracy_chart():
    """Creates a bar chart showing...
    """
    print('Test')


def surfline_binary_precision_chart():
    """Creates a bar chart showing...
    """
    print('Test')


def surfline_binary_recall_chart():
    """Creates a bar chart showing...
    """
    print('Test')


def main():
    magicseaweed_binary_accuracy_chart()


if __name__ == '__main__':
    main()
