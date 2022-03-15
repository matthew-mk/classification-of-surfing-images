"""This module contains functions that create graphs to show the results that were gathered for the binary
classification models during the evaluation stage of the project. """

from utils.helper_utils import create_binary_bar_chart

def magicseaweed_binary_accuracy_chart():
    """Creates a bar chart showing...
    """
    cnn_results = [100, 99, 99, 99, 99]
    svm_results = [98, 96, 94, 95, 94]
    rf_results = [97, 95, 93, 94, 94]
    knn_results = [78, 74, 78, 76, 77]
    create_binary_bar_chart(cnn_results, svm_results, rf_results, knn_results)


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
