"""This module creates charts that show the k-fold cross validation results that were gathered during the evaluation
stage of the project. K-fold cross validation was applied to the models using the binary and rating datasets from
Magicseaweed and Surfline that contained images from 5 surfing locations. """

from utils.helper_utils import create_evaluation_bar_chart


def magicseaweed_binary_kfold_results_chart():
    """Creates a bar chart showing...
    """
    cnn_results = [95, 94, 96]
    svm_results = [90, 93, 87]
    rf_results = [88, 92, 83]
    knn_results = [64, 95, 29]
    title = '5-Fold Cross Validation Average Results on Magicseaweed Binary Datasets (5 locations)'
    x_labels = ['Accuracy', 'Precision', 'Recall']
    x_axis_label = 'Performance metrics'
    y_axis_label = 'Percentage'
    create_evaluation_bar_chart(cnn_results, svm_results, rf_results, knn_results, title, x_labels, x_axis_label,
                                y_axis_label)


def surfline_binary_kfold_results_chart():
    """Creates a bar chart showing...
    """
    cnn_results = [94, 91, 97]
    svm_results = [89, 92, 86]
    rf_results = [86, 86, 86]
    knn_results = [67, 90, 39]
    title = '5-Fold Cross Validation Average Results on Surfline Binary Datasets (5 locations)'
    x_labels = ['Accuracy', 'Precision', 'Recall']
    x_axis_label = 'Performance metrics'
    y_axis_label = 'Percentage'
    create_evaluation_bar_chart(cnn_results, svm_results, rf_results, knn_results, title, x_labels, x_axis_label,
                                y_axis_label)


def magicseaweed_rating_kfold_accuracy_chart():
    """Creates a bar chart showing...
    """
    cnn_results = [63]
    svm_results = [58]
    rf_results = [57]
    knn_results = [35]
    title = '5-Fold Cross Validation Average Accuracy on Magicseaweed Rating Datasets (5 locations)'
    x_labels = ['Accuracy']
    x_axis_label = 'Performance metric'
    y_axis_label = 'Percentage'
    create_evaluation_bar_chart(cnn_results, svm_results, rf_results, knn_results, title, x_labels, x_axis_label,
                                y_axis_label)


def magicseaweed_rating_kfold_precision_chart():
    """Creates a bar chart showing...
    """
    cnn_results = [74, 70, 50, 60, 4]
    svm_results = [56, 60, 54, 61, 0]
    rf_results = [53, 68, 38, 62, 0]
    knn_results = [28, 46, 3, 66, 0]
    title = '5-Fold Cross Validation Average Precision on Magicseaweed Rating Datasets (5 locations)'
    x_labels = ['Flat', 'Poor', 'Fair', 'Good', 'Epic']
    x_axis_label = 'Rating'
    y_axis_label = 'Precision'
    create_evaluation_bar_chart(cnn_results, svm_results, rf_results, knn_results, title, x_labels, x_axis_label,
                                y_axis_label)


def magicseaweed_rating_kfold_recall_chart():
    """Creates a bar chart showing...
    """
    cnn_results = [65, 70, 49, 70, 5]
    svm_results = [64, 66, 35, 66, 0]
    rf_results = [72, 56, 28, 69, 0]
    knn_results = [70, 58, 1, 6, 0]
    title = '5-Fold Cross Validation Average Recall on Magicseaweed Rating Datasets (5 locations)'
    x_labels = ['Flat', 'Poor', 'Fair', 'Good', 'Epic']
    x_axis_label = 'Rating'
    y_axis_label = 'Recall'
    create_evaluation_bar_chart(cnn_results, svm_results, rf_results, knn_results, title, x_labels, x_axis_label,
                                y_axis_label)


def surfline_rating_kfold_accuracy_chart():
    """Creates a bar chart showing...
    """
    cnn_results = [62]
    svm_results = [62]
    rf_results = [57]
    knn_results = [39]
    title = '5-Fold Cross Validation Average Accuracy on Surfline Rating Datasets (5 locations)'
    x_labels = ['Accuracy']
    x_axis_label = 'Performance metric'
    y_axis_label = 'Percentage'
    create_evaluation_bar_chart(cnn_results, svm_results, rf_results, knn_results, title, x_labels, x_axis_label,
                                y_axis_label)


def surfline_rating_kfold_precision_chart():
    """Creates a bar chart showing...
    """
    cnn_results = [69, 54, 68, 52, 75]
    svm_results = [59, 53, 71, 66, 50]
    rf_results = [64, 42, 60, 48, 20]
    knn_results = [39, 31, 62, 51, 0]
    title = '5-Fold Cross Validation Average Precision on Surfline Rating Datasets (5 locations)'
    x_labels = ['Flat', 'Poor', 'Fair', 'Good', 'Epic']
    x_axis_label = 'Rating'
    y_axis_label = 'Precision'
    create_evaluation_bar_chart(cnn_results, svm_results, rf_results, knn_results, title, x_labels, x_axis_label,
                                y_axis_label)


def surfline_rating_kfold_recall_chart():
    """Creates a bar chart showing...
    """
    cnn_results = [63, 49, 74, 59, 60]
    svm_results = [76, 46, 71, 53, 20]
    rf_results = [79, 33, 72, 28, 7]
    knn_results = [88, 35, 10, 16, 0]
    title = '5-Fold Cross Validation Average Recall on Surfline Rating Datasets (5 locations)'
    x_labels = ['Flat', 'Poor', 'Fair', 'Good', 'Epic']
    x_axis_label = 'Rating'
    y_axis_label = 'Recall'
    create_evaluation_bar_chart(cnn_results, svm_results, rf_results, knn_results, title, x_labels, x_axis_label,
                                y_axis_label)


def main():
    # 5-Fold Cross Validation Average Results on Magicseaweed Binary Datasets (5 locations)
    magicseaweed_binary_kfold_results_chart()

    # 5-Fold Cross Validation Average Results on Surfline Binary Datasets (5 locations)
    surfline_binary_kfold_results_chart()

    # 5-Fold Cross Validation Average Accuracy on Magicseaweed Rating Datasets (5 locations)
    magicseaweed_rating_kfold_accuracy_chart()

    # 5-Fold Cross Validation Average Precision on Magicseaweed Rating Datasets (5 locations)
    # magicseaweed_rating_kfold_precision_chart()

    # 5-Fold Cross Validation Average Recall on Magicseaweed Rating Datasets (5 locations)
    # magicseaweed_rating_kfold_recall_chart()

    # 5-Fold Cross Validation Average Accuracy on Surfline Rating Datasets (5 locations)
    surfline_rating_kfold_accuracy_chart()

    # 5-Fold Cross Validation Average Precision on Surfline Rating Datasets (5 locations)
    # surfline_rating_kfold_precision_chart()

    # 5-Fold Cross Validation Average Recall on Surfline Rating Datasets (5 locations)
    # surfline_rating_kfold_recall_chart()


if __name__ == '__main__':
    main()
