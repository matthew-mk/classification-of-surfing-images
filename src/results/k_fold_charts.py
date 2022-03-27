"""This module creates charts that show the k-fold cross validation results that were gathered during the evaluation
stage of the project. K-fold cross validation was applied to the models using the binary and rating datasets from
Magicseaweed and Surfline that contained images from 5 surfing locations. """

from utils.helper_utils import create_evaluation_bar_chart_1, create_evaluation_bar_chart_2


def magicseaweed_binary_kfold_results_chart():
    """A dataset was created that contained images from 5 different surfing locations from Magicseaweed. Each image was
    classed as 'surfable' or 'unsurfable' depending on whether the conditions in the images were suitable for surfing or
    not. This function creates a bar chart showing the average accuracy, precision and recall that the CNN, SVM, RF, and
    KNN models achieved when 5-fold cross validation was applied to them using this dataset.
    """
    cnn_results = [95, 94, 96]
    svm_results = [90, 93, 87]
    rf_results = [88, 92, 83]
    knn_results = [64, 95, 29]
    title = '5-Fold Cross Validation Average Results on Magicseaweed Binary Datasets (5 locations)'
    x_labels = ['Accuracy', 'Precision', 'Recall']
    x_axis_label = 'Performance metrics'
    y_axis_label = 'Percentage'
    create_evaluation_bar_chart_1(cnn_results, svm_results, rf_results, knn_results, title, x_labels, x_axis_label,
                                  y_axis_label)


def surfline_binary_kfold_results_chart():
    """A dataset was created that contained images from 5 different surfing locations from Surfline. Each image was
    classed as 'surfable' or 'unsurfable' depending on whether the conditions in the images were suitable for surfing or
    not. This function creates a bar chart showing the average accuracy, precision and recall that the CNN, SVM, RF, and
    KNN models achieved when 5-fold cross validation was applied to them using this dataset.
    """
    cnn_results = [94, 91, 97]
    svm_results = [89, 92, 86]
    rf_results = [86, 86, 86]
    knn_results = [67, 90, 39]
    title = '5-Fold Cross Validation Average Results on Surfline Binary Datasets (5 locations)'
    x_labels = ['Accuracy', 'Precision', 'Recall']
    x_axis_label = 'Performance metrics'
    y_axis_label = 'Percentage'
    create_evaluation_bar_chart_1(cnn_results, svm_results, rf_results, knn_results, title, x_labels, x_axis_label,
                                  y_axis_label)


def magicseaweed_rating_kfold_accuracy_chart():
    """A dataset was created that contained images from 5 different surfing locations from Magicseaweed. Each image was
    classed as 'flat', 'poor', 'fair', 'good', or 'epic' depending on how good the conditions in the images were for
    surfing. This function creates a bar chart showing the average accuracy the CNN, SVM, RF, and KNN models achieved
    when 5-fold cross validation was applied to them using this dataset.
    """
    results = [63, 58, 57, 35]
    title = '5-Fold Cross Validation Average Accuracy on Magicseaweed Rating Datasets (5 locations)'
    y_axis_label = 'Accuracy (%)'
    create_evaluation_bar_chart_2(results, title, y_axis_label)


def magicseaweed_rating_kfold_precision_chart():
    """A dataset was created that contained images from 5 different surfing locations from Magicseaweed. Each image
    was classed as 'flat', 'poor', 'fair', 'good', or 'epic' depending on how good the conditions in the images were
    for surfing. This function creates a bar chart showing the average precision the CNN, SVM, RF, and KNN models
    achieved when 5-fold cross validation was applied to them using this dataset. The average precision is shown for
    each category (flat, poor, fair, good, and epic).
    """
    cnn_results = [74, 70, 50, 60, 4]
    svm_results = [56, 60, 54, 61, 0]
    rf_results = [53, 68, 38, 62, 0]
    knn_results = [28, 46, 3, 66, 0]
    title = '5-Fold Cross Validation Average Precision on Magicseaweed Rating Datasets (5 locations)'
    x_labels = ['Flat', 'Poor', 'Fair', 'Good', 'Epic']
    x_axis_label = 'Category'
    y_axis_label = 'Precision'
    create_evaluation_bar_chart_1(cnn_results, svm_results, rf_results, knn_results, title, x_labels, x_axis_label,
                                  y_axis_label)


def magicseaweed_rating_kfold_recall_chart():
    """A dataset was created that contained images from 5 different surfing locations from Magicseaweed. Each image was
    classed as 'flat', 'poor', 'fair', 'good', or 'epic' depending on how good the conditions in the images were for
    surfing. This function creates a bar chart showing the average recall the CNN, SVM, RF, and KNN models achieved
    when 5-fold cross validation was applied to them using this dataset. The average recall is shown for each category
    (flat, poor, fair, good, and epic).
    """
    cnn_results = [65, 70, 49, 70, 5]
    svm_results = [64, 66, 35, 66, 0]
    rf_results = [72, 56, 28, 69, 0]
    knn_results = [70, 58, 1, 6, 0]
    title = '5-Fold Cross Validation Average Recall on Magicseaweed Rating Datasets (5 locations)'
    x_labels = ['Flat', 'Poor', 'Fair', 'Good', 'Epic']
    x_axis_label = 'Category'
    y_axis_label = 'Recall'
    create_evaluation_bar_chart_1(cnn_results, svm_results, rf_results, knn_results, title, x_labels, x_axis_label,
                                  y_axis_label)


def surfline_rating_kfold_accuracy_chart():
    """A dataset was created that contained images from 5 different surfing locations from Surfline. Each image was
    classed as 'flat', 'poor', 'fair', 'good', or 'epic' depending on how good the conditions in the images were for
    surfing. This function creates a bar chart showing the average accuracy the CNN, SVM, RF, and KNN models achieved
    when 5-fold cross validation was applied to them using this dataset.
    """

    title = '5-Fold Cross Validation Average Accuracy on Surfline Rating Datasets (5 locations)'
    results = [62, 62, 57, 39]
    y_axis_label = 'Accuracy (%)'
    create_evaluation_bar_chart_2(results, title, y_axis_label)


def surfline_rating_kfold_precision_chart():
    """A dataset was created that contained images from 5 different surfing locations from Surfline. Each image was
    classed as 'flat', 'poor', 'fair', 'good', or 'epic' depending on how good the conditions in the images were for
    surfing. This function creates a bar chart showing the average precision the CNN, SVM, RF, and KNN models
    achieved when 5-fold cross validation was applied to them using this dataset. The average precision is shown for
    each category (flat, poor, fair, good, and epic).
    """
    cnn_results = [69, 54, 68, 52, 75]
    svm_results = [59, 53, 71, 66, 50]
    rf_results = [64, 42, 60, 48, 20]
    knn_results = [39, 31, 62, 51, 0]
    title = '5-Fold Cross Validation Average Precision on Surfline Rating Datasets (5 locations)'
    x_labels = ['Flat', 'Poor', 'Fair', 'Good', 'Epic']
    x_axis_label = 'Category'
    y_axis_label = 'Precision'
    create_evaluation_bar_chart_1(cnn_results, svm_results, rf_results, knn_results, title, x_labels, x_axis_label,
                                  y_axis_label)


def surfline_rating_kfold_recall_chart():
    """A dataset was created that contained images from 5 different surfing locations from Surfline. Each image was
    classed as 'flat', 'poor', 'fair', 'good', or 'epic' depending on how good the conditions in the images were for
    surfing. This function creates a bar chart showing the average recall the CNN, SVM, RF, and KNN models achieved
    when 5-fold cross validation was applied to them using this dataset. The average recall is shown for each category
    (flat, poor, fair, good, and epic).
    """
    cnn_results = [63, 49, 74, 59, 60]
    svm_results = [76, 46, 71, 53, 20]
    rf_results = [79, 33, 72, 28, 7]
    knn_results = [88, 35, 10, 16, 0]
    title = '5-Fold Cross Validation Average Recall on Surfline Rating Datasets (5 locations)'
    x_labels = ['Flat', 'Poor', 'Fair', 'Good', 'Epic']
    x_axis_label = 'Category'
    y_axis_label = 'Recall'
    create_evaluation_bar_chart_1(cnn_results, svm_results, rf_results, knn_results, title, x_labels, x_axis_label,
                                  y_axis_label)


def main():
    # 5-Fold Cross Validation Average Results on Magicseaweed Binary Datasets (5 locations)
    magicseaweed_binary_kfold_results_chart()

    # 5-Fold Cross Validation Average Results on Surfline Binary Datasets (5 locations)
    # surfline_binary_kfold_results_chart()

    # 5-Fold Cross Validation Average Accuracy on Magicseaweed Rating Datasets (5 locations)
    # magicseaweed_rating_kfold_accuracy_chart()

    # 5-Fold Cross Validation Average Precision on Magicseaweed Rating Datasets (5 locations)
    # magicseaweed_rating_kfold_precision_chart()

    # 5-Fold Cross Validation Average Recall on Magicseaweed Rating Datasets (5 locations)
    # magicseaweed_rating_kfold_recall_chart()

    # 5-Fold Cross Validation Average Accuracy on Surfline Rating Datasets (5 locations)
    # surfline_rating_kfold_accuracy_chart()

    # 5-Fold Cross Validation Average Precision on Surfline Rating Datasets (5 locations)
    # surfline_rating_kfold_precision_chart()

    # 5-Fold Cross Validation Average Recall on Surfline Rating Datasets (5 locations)
    # surfline_rating_kfold_recall_chart()


if __name__ == '__main__':
    main()
