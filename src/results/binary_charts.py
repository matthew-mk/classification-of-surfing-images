"""This module contains functions that create graphs to show the results that were gathered for the binary
classification models during the evaluation stage of the project. """

from utils.helper_utils import create_evaluation_bar_chart

def magicseaweed_binary_accuracy_chart():
    """Creates a bar chart showing...
    """
    cnn_results = [100, 99, 99, 99, 99]
    svm_results = [98, 96, 94, 95, 94]
    rf_results = [97, 95, 93, 94, 94]
    knn_results = [78, 74, 78, 76, 77]
    title = 'Accuracy of Binary Models on Magicseaweed Datasets (Best Seeds)'
    y_axis_label = 'Accuracy (%)'
    create_evaluation_bar_chart(cnn_results, svm_results, rf_results, knn_results, title, y_axis_label)


def magicseaweed_binary_precision_chart():
    """Creates a bar chart showing...
    """
    cnn_results = [100, 100, 98, 98, 97]
    svm_results = [100, 97, 94, 98, 97]
    rf_results = [97, 98, 98, 100, 96]
    knn_results = [100, 94, 100, 94, 97]
    title = 'Precision of Binary Models on Magicseaweed Datasets (Best Seeds)'
    y_axis_label = 'Precision'
    create_evaluation_bar_chart(cnn_results, svm_results, rf_results, knn_results, title, y_axis_label)


def magicseaweed_binary_recall_chart():
    """Creates a bar chart showing...
    """
    cnn_results = [100, 97, 100, 100, 100]
    svm_results = [96, 95, 94, 91, 91]
    rf_results = [97, 93, 88, 87, 93]
    knn_results = [46, 44, 46, 36, 48]
    title = 'Recall of Binary Models on Magicseaweed Datasets (Best Seeds)'
    y_axis_label = 'Recall'
    create_evaluation_bar_chart(cnn_results, svm_results, rf_results, knn_results, title, y_axis_label)


def surfline_binary_accuracy_chart():
    """Creates a bar chart showing...
    """
    cnn_results = [100, 100, 99, 98, 97]
    svm_results = [97, 95, 93, 94, 93]
    rf_results = [95, 95, 91, 94, 92]
    knn_results = [92, 80, 81, 80, 76]
    title = 'Accuracy of Binary Models on Surfline Datasets (Best Seeds)'
    y_axis_label = 'Accuracy (%)'
    create_evaluation_bar_chart(cnn_results, svm_results, rf_results, knn_results, title, y_axis_label)


def surfline_binary_precision_chart():
    """Creates a bar chart showing...
    """
    cnn_results = [100, 100, 100, 97, 97]
    svm_results = [97, 94, 98, 95, 92]
    rf_results = [93, 94, 94, 92, 92]
    knn_results = [100, 95, 100, 94, 95]
    title = 'Precision of Binary Models on Surfline Datasets (Best Seeds)'
    y_axis_label = 'Precision'
    create_evaluation_bar_chart(cnn_results, svm_results, rf_results, knn_results, title, y_axis_label)


def surfline_binary_recall_chart():
    """Creates a bar chart showing...
    """
    cnn_results = [100, 100, 98, 98, 97]
    svm_results = [97, 94, 89, 93, 94]
    rf_results = [96, 94, 88, 97, 93]
    knn_results = [81, 56, 59, 58, 54]
    title = 'Recall of Binary Models on Surfline Datasets (Best Seeds)'
    y_axis_label = 'Recall'
    create_evaluation_bar_chart(cnn_results, svm_results, rf_results, knn_results, title, y_axis_label)


def main():
    # Accuracy of Binary Models on Magicseaweed Datasets (Best Seeds)
    magicseaweed_binary_accuracy_chart()

    # Precision of Binary Models on Magicseaweed Datasets (Best Seeds)
    # magicseaweed_binary_precision_chart()

    # Recall of Binary Models on Magicseaweed Datasets (Best Seeds)
    # magicseaweed_binary_recall_chart()

    # Accuracy of Binary Models on Surfline Datasets (Best Seeds)
    # surfline_binary_accuracy_chart()

    # Precision of Binary Models on Surfline Datasets (Best Seeds)
    # surfline_binary_precision_chart()

    # Recall of Binary Models on Surfline Datasets (Best Seeds)
    # surfline_binary_recall_chart()


if __name__ == '__main__':
    main()
