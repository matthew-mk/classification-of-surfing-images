"""This module creates charts that show the results of training and testing the models on the best seeds that were found
of the Magicseaweed and Surfline datasets. """

from utils.helper_utils import create_evaluation_bar_chart


def magicseaweed_binary_accuracy_chart():
    """Creates a bar chart showing...
    """
    cnn_results = [100, 99, 99, 99, 99]
    svm_results = [98, 96, 94, 95, 94]
    rf_results = [97, 95, 93, 94, 94]
    knn_results = [78, 74, 78, 76, 77]
    title = 'Accuracy of Models on Best Seeds of Magicseaweed Binary Datasets'
    x_labels = ['1', '2', '3', '4', '5']
    x_axis_label = 'Number of locations'
    y_axis_label = 'Accuracy (%)'
    create_evaluation_bar_chart(cnn_results, svm_results, rf_results, knn_results, title, x_labels, x_axis_label,
                                y_axis_label)


def magicseaweed_binary_precision_chart():
    """Creates a bar chart showing...
    """
    cnn_results = [100, 100, 98, 98, 97]
    svm_results = [100, 97, 94, 98, 97]
    rf_results = [97, 98, 98, 100, 96]
    knn_results = [100, 94, 100, 94, 97]
    title = 'Precision of Models on Best Seeds of Magicseaweed Binary Datasets'
    x_labels = ['1', '2', '3', '4', '5']
    x_axis_label = 'Number of locations'
    y_axis_label = 'Precision'
    create_evaluation_bar_chart(cnn_results, svm_results, rf_results, knn_results, title, x_labels, x_axis_label,
                                y_axis_label)


def magicseaweed_binary_recall_chart():
    """Creates a bar chart showing...
    """
    cnn_results = [100, 97, 100, 100, 100]
    svm_results = [96, 95, 94, 91, 91]
    rf_results = [97, 93, 88, 87, 93]
    knn_results = [46, 44, 46, 36, 48]
    title = 'Recall of Models on Best Seeds of Magicseaweed Binary Datasets'
    x_labels = ['1', '2', '3', '4', '5']
    x_axis_label = 'Number of locations'
    y_axis_label = 'Recall'
    create_evaluation_bar_chart(cnn_results, svm_results, rf_results, knn_results, title, x_labels, x_axis_label,
                                y_axis_label)


def surfline_binary_accuracy_chart():
    """Creates a bar chart showing...
    """
    cnn_results = [100, 100, 99, 98, 97]
    svm_results = [97, 95, 93, 94, 93]
    rf_results = [95, 95, 91, 94, 92]
    knn_results = [92, 80, 81, 80, 76]
    title = 'Accuracy of Models on Best Seeds of Surfline Binary Datasets'
    x_labels = ['1', '2', '3', '4', '5']
    x_axis_label = 'Number of locations'
    y_axis_label = 'Accuracy (%)'
    create_evaluation_bar_chart(cnn_results, svm_results, rf_results, knn_results, title, x_labels, x_axis_label,
                                y_axis_label)


def surfline_binary_precision_chart():
    """Creates a bar chart showing...
    """
    cnn_results = [100, 100, 100, 97, 97]
    svm_results = [97, 94, 98, 95, 92]
    rf_results = [93, 94, 94, 92, 92]
    knn_results = [100, 95, 100, 94, 95]
    title = 'Precision of Models on Best Seeds of Surfline Binary Datasets'
    x_labels = ['1', '2', '3', '4', '5']
    x_axis_label = 'Number of locations'
    y_axis_label = 'Precision'
    create_evaluation_bar_chart(cnn_results, svm_results, rf_results, knn_results, title, x_labels, x_axis_label,
                                y_axis_label)


def surfline_binary_recall_chart():
    """Creates a bar chart showing...
    """
    cnn_results = [100, 100, 98, 98, 97]
    svm_results = [97, 94, 89, 93, 94]
    rf_results = [96, 94, 88, 97, 93]
    knn_results = [81, 56, 59, 58, 54]
    title = 'Recall of Models on Best Seeds of Surfline Binary Datasets'
    x_labels = ['1', '2', '3', '4', '5']
    x_axis_label = 'Number of locations'
    y_axis_label = 'Recall'
    create_evaluation_bar_chart(cnn_results, svm_results, rf_results, knn_results, title, x_labels, x_axis_label,
                                y_axis_label)


def magicseaweed_rating_accuracy_chart():
    """Creates a bar chart showing...
    """
    cnn_results = [80, 73, 65, 64, 64]
    svm_results = [85, 71, 67, 64, 63]
    rf_results = [78, 69, 62, 63, 66]
    knn_results = [52, 50, 43, 45, 44]
    title = 'Accuracy of Models on Best Seeds of Magicseaweed Rating Datasets'
    x_labels = ['1', '2', '3', '4', '5']
    x_axis_label = 'Number of locations'
    y_axis_label = 'Accuracy (%)'
    create_evaluation_bar_chart(cnn_results, svm_results, rf_results, knn_results, title, x_labels, x_axis_label,
                                y_axis_label)


def surfline_rating_accuracy_chart():
    """Creates a bar chart showing...
    """
    cnn_results = [62, 65, 79, 73, 74]
    svm_results = [63, 65, 72, 68, 71]
    rf_results = [62, 64, 74, 70, 73]
    knn_results = [48, 44, 47, 46, 47]
    title = 'Accuracy of Models on Best Seeds of Surfline Rating Datasets'
    x_labels = ['1', '2', '3', '4', '5']
    x_axis_label = 'Number of locations'
    y_axis_label = 'Accuracy (%)'
    create_evaluation_bar_chart(cnn_results, svm_results, rf_results, knn_results, title, x_labels, x_axis_label,
                                y_axis_label)


def main():
    # Accuracy of Models on Best Seeds of Magicseaweed Binary Datasets
    magicseaweed_binary_accuracy_chart()

    # Precision of Models on Best Seeds of Magicseaweed Binary Datasets
    # magicseaweed_binary_precision_chart()

    # Recall of Models on Best Seeds of Magicseaweed Binary Datasets
    # magicseaweed_binary_recall_chart()

    # Accuracy of Models on Best Seeds of Surfline Binary Datasets
    # surfline_binary_accuracy_chart()

    # Precision of Models on Best Seeds of Surfline Binary Datasets
    # surfline_binary_precision_chart()

    # Recall of Models on Best Seeds of Surfline Binary Datasets
    # surfline_binary_recall_chart()

    # Accuracy of Models on Best Seeds of Magicseaweed Rating Datasets
    # magicseaweed_rating_accuracy_chart()

    # Accuracy of Models on Best Seeds of Surfline Rating Datasets
    # surfline_rating_accuracy_chart()


if __name__ == '__main__':
    main()
