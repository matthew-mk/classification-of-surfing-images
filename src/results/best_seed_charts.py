"""This module creates charts that show the results of training and testing the models on the best seeds that were found
of the Magicseaweed and Surfline datasets. """

from utils.helper_utils import create_evaluation_bar_chart


def magicseaweed_binary_accuracy_chart():
    """The binary CNN, SVM, RF, and KNN models were trained and tested on datasets that contained images of up to 5
    surfing locations from Magicseaweed. Each image was classed as 'surfable' or 'unsurfable' depending on whether
    the conditions in the images were suitable for surfing or not. This function creates a bar chart showing the
    accuracy the models achieved on the best seeds that were found for each of these datasets.
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
    """The binary CNN, SVM, RF, and KNN models were trained and tested on datasets that contained images of up to 5
    surfing locations from Magicseaweed. Each image was classed as 'surfable' or 'unsurfable' depending on whether
    the conditions in the images were suitable for surfing or not. This function creates a bar chart showing the
    precision the models achieved on the best seeds that were found for each of these datasets.
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
    """The binary CNN, SVM, RF, and KNN models were trained and tested on datasets that contained images of up to 5
    surfing locations from Magicseaweed. Each image was classed as 'surfable' or 'unsurfable' depending on whether
    the conditions in the images were suitable for surfing or not. This function creates a bar chart showing the recall
    the models achieved on the best seeds that were found for each of these datasets.
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
    """The binary CNN, SVM, RF, and KNN models were trained and tested on datasets that contained images of up to 5
    surfing locations from Surfline. Each image was classed as 'surfable' or 'unsurfable' depending on whether
    the conditions in the images were suitable for surfing or not. This function creates a bar chart showing the
    accuracy the models achieved on the best seeds that were found for each of these datasets.
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
    """The binary CNN, SVM, RF, and KNN models were trained and tested on datasets that contained images of up to 5
    surfing locations from Surfline. Each image was classed as 'surfable' or 'unsurfable' depending on whether
    the conditions in the images were suitable for surfing or not. This function creates a bar chart showing the
    precision the models achieved on the best seeds that were found for each of these datasets.
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
    """The binary CNN, SVM, RF, and KNN models were trained and tested on datasets that contained images of up to 5
    surfing locations from Surfline. Each image was classed as 'surfable' or 'unsurfable' depending on whether
    the conditions in the images were suitable for surfing or not. This function creates a bar chart showing the recall
    the models achieved on the best seeds that were found for each of these datasets.
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
    """CNN, SVM, RF, and KNN models were trained and tested on datasets that contained images of up to 5 surfing
    locations from Magicseaweed. Each image was classed as 'flat', 'poor', 'fair', 'good', or 'epic' depending on how
    good the conditions in the images were for surfing. This function creates a bar chart showing the accuracy the
    models achieved on the best seeds that were found for each of these datasets.
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


def magicseaweed_rating_precision_1_chart():
    """CNN, SVM, RF, and KNN models were trained and tested on datasets that contained images of up to 5 surfing
    locations from Magicseaweed. Each image was classed as 'flat', 'poor', 'fair', 'good', or 'epic' depending on how
    good the conditions in the images were for surfing. This function creates a bar chart showing the precision the
    models achieved on the best seed of the dataset that contained images from 1 surfing location (Bantham Beach).
    """
    cnn_results = [80, 89, 79, 67, 50]
    svm_results = [80, 96, 83, 71, 0]
    rf_results = [50, 85, 77, 85, 0]
    knn_results = [12, 55, 100, 0, 0]
    title = 'Precision of Models on Best Seeds of Magicseaweed Rating Datasets (1 location)'
    x_labels = ['Flat', 'Poor', 'Fair', 'Good', 'Epic']
    x_axis_label = 'Category'
    y_axis_label = 'Precision'
    create_evaluation_bar_chart(cnn_results, svm_results, rf_results, knn_results, title, x_labels, x_axis_label,
                                y_axis_label)


def magicseaweed_rating_precision_2_chart():
    """CNN, SVM, RF, and KNN models were trained and tested on datasets that contained images of up to 5 surfing
    locations from Magicseaweed. Each image was classed as 'flat', 'poor', 'fair', 'good', or 'epic' depending on how
    good the conditions in the images were for surfing. This function creates a bar chart showing the precision the
    models achieved on the best seed of the dataset that contained images from 2 surfing locations (Bantham Beach and
    Polzeath).
    """
    cnn_results = [77, 85, 59, 58, 0]
    svm_results = [67, 79, 57, 71, 0]
    rf_results = [56, 81, 73, 56, 0]
    knn_results = [30, 68, 60, 0, 0]
    title = 'Precision of Models on Best Seeds of Magicseaweed Rating Datasets (2 locations)'
    x_labels = ['Flat', 'Poor', 'Fair', 'Good', 'Epic']
    x_axis_label = 'Category'
    y_axis_label = 'Precision'
    create_evaluation_bar_chart(cnn_results, svm_results, rf_results, knn_results, title, x_labels, x_axis_label,
                                y_axis_label)


def magicseaweed_rating_precision_3_chart():
    """CNN, SVM, RF, and KNN models were trained and tested on datasets that contained images of up to 5 surfing
    locations from Magicseaweed. Each image was classed as 'flat', 'poor', 'fair', 'good', or 'epic' depending on how
    good the conditions in the images were for surfing. This function creates a bar chart showing the precision the
    models achieved on the best seed of the dataset that contained images from 3 surfing locations (Bantham Beach,
    Polzeath, and Porthowan).
    """
    cnn_results = [88, 69, 44, 70, 0]
    svm_results = [60, 73, 45, 75, 33]
    rf_results = [73, 55, 50, 68, 0]
    knn_results = [33, 52, 40, 100, 0]
    title = 'Precision of Models on Best Seeds of Magicseaweed Rating Datasets (3 locations)'
    x_labels = ['Flat', 'Poor', 'Fair', 'Good', 'Epic']
    x_axis_label = 'Category'
    y_axis_label = 'Precision'
    create_evaluation_bar_chart(cnn_results, svm_results, rf_results, knn_results, title, x_labels, x_axis_label,
                                y_axis_label)


def magicseaweed_rating_precision_4_chart():
    """CNN, SVM, RF, and KNN models were trained and tested on datasets that contained images of up to 5 surfing
    locations from Magicseaweed. Each image was classed as 'flat', 'poor', 'fair', 'good', or 'epic' depending on how
    good the conditions in the images were for surfing. This function creates a bar chart showing the precision the
    models achieved on the best seed of the dataset that contained images from 4 surfing locations (Bantham Beach,
    Polzeath, Porthowan, and Praa Sands).
    """
    cnn_results = [70, 63, 63, 67, 0]
    svm_results = [73, 63, 62, 60, 0]
    rf_results = [54, 73, 60, 68, 0]
    knn_results = [35, 68, 50, 80, 0]
    title = 'Precision of Models on Best Seeds of Magicseaweed Rating Datasets (4 locations)'
    x_labels = ['Flat', 'Poor', 'Fair', 'Good', 'Epic']
    x_axis_label = 'Category'
    y_axis_label = 'Precision'
    create_evaluation_bar_chart(cnn_results, svm_results, rf_results, knn_results, title, x_labels, x_axis_label,
                                y_axis_label)


def magicseaweed_rating_precision_5_chart():
    """CNN, SVM, RF, and KNN models were trained and tested on datasets that contained images of up to 5 surfing
    locations from Magicseaweed. Each image was classed as 'flat', 'poor', 'fair', 'good', or 'epic' depending on how
    good the conditions in the images were for surfing. This function creates a bar chart showing the precision the
    models achieved on the best seed of the dataset that contained images from all 5 surfing locations (Bantham Beach,
    Polzeath, Porthowan, Praa Sands, and Widemouth Bay).
    """
    cnn_results = [77, 80, 44, 56, 0]
    svm_results = [60, 67, 50, 69, 0]
    rf_results = [56, 73, 73, 67, 0]
    knn_results = [32, 55, 33, 100, 0]
    title = 'Precision of Models on Best Seeds of Magicseaweed Rating Datasets (5 locations)'
    x_labels = ['Flat', 'Poor', 'Fair', 'Good', 'Epic']
    x_axis_label = 'Category'
    y_axis_label = 'Precision'
    create_evaluation_bar_chart(cnn_results, svm_results, rf_results, knn_results, title, x_labels, x_axis_label,
                                y_axis_label)


def magicseaweed_rating_recall_1_chart():
    """CNN, SVM, RF, and KNN models were trained and tested on datasets that contained images of up to 5 surfing
    locations from Magicseaweed. Each image was classed as 'flat', 'poor', 'fair', 'good', or 'epic' depending on how
    good the conditions in the images were for surfing. This function creates a bar chart showing the recall the
    models achieved on the best seed of the dataset that contained images from 1 surfing location (Bantham Beach).
    """
    cnn_results = [67, 96, 73, 67, 50]
    svm_results = [100, 100, 79, 77, 0]
    rf_results = [67, 88, 67, 92, 0]
    knn_results = [12, 100, 30, 0, 0]
    title = 'Recall of Models on Best Seeds of Magicseaweed Rating Datasets (1 location)'
    x_labels = ['Flat', 'Poor', 'Fair', 'Good', 'Epic']
    x_axis_label = 'Category'
    y_axis_label = 'Recall'
    create_evaluation_bar_chart(cnn_results, svm_results, rf_results, knn_results, title, x_labels, x_axis_label,
                                y_axis_label)


def magicseaweed_rating_recall_2_chart():
    """CNN, SVM, RF, and KNN models were trained and tested on datasets that contained images of up to 5 surfing
    locations from Magicseaweed. Each image was classed as 'flat', 'poor', 'fair', 'good', or 'epic' depending on how
    good the conditions in the images were for surfing. This function creates a bar chart showing the recall the
    models achieved on the best seed of the dataset that contained images from 2 surfing locations (Bantham Beach and
    Polzeath).
    """
    cnn_results = [62, 90, 76, 44, 0]
    svm_results = [80, 84, 50, 62, 0]
    rf_results = [62, 81, 65, 56, 0]
    knn_results = [73, 81, 19, 0, 0]
    title = 'Recall of Models on Best Seeds of Magicseaweed Rating Datasets (2 locations)'
    x_labels = ['Flat', 'Poor', 'Fair', 'Good', 'Epic']
    x_axis_label = 'Category'
    y_axis_label = 'Recall'
    create_evaluation_bar_chart(cnn_results, svm_results, rf_results, knn_results, title, x_labels, x_axis_label,
                                y_axis_label)


def magicseaweed_rating_recall_3_chart():
    """CNN, SVM, RF, and KNN models were trained and tested on datasets that contained images of up to 5 surfing
    locations from Magicseaweed. Each image was classed as 'flat', 'poor', 'fair', 'good', or 'epic' depending on how
    good the conditions in the images were for surfing. This function creates a bar chart showing the recall the
    models achieved on the best seed of the dataset that contained images from 3 surfing locations (Bantham Beach,
    Polzeath, and Porthowan).
    """
    cnn_results = [52, 89, 61, 64, 0]
    svm_results = [75, 76, 28, 75, 100]
    rf_results = [59, 78, 33, 76, 0]
    knn_results = [62, 83, 10, 5, 0]
    title = 'Recall of Models on Best Seeds of Magicseaweed Rating Datasets (3 locations)'
    x_labels = ['Flat', 'Poor', 'Fair', 'Good', 'Epic']
    x_axis_label = 'Category'
    y_axis_label = 'Recall'
    create_evaluation_bar_chart(cnn_results, svm_results, rf_results, knn_results, title, x_labels, x_axis_label,
                                y_axis_label)

def magicseaweed_rating_recall_4_chart():
    """CNN, SVM, RF, and KNN models were trained and tested on datasets that contained images of up to 5 surfing
    locations from Magicseaweed. Each image was classed as 'flat', 'poor', 'fair', 'good', or 'epic' depending on how
    good the conditions in the images were for surfing. This function creates a bar chart showing the recall the
    models achieved on the best seed of the dataset that contained images from 4 surfing locations (Bantham Beach,
    Polzeath, Porthowan, and Praa Sands).
    """
    cnn_results = [66, 66, 74, 61, 0]
    svm_results = [69, 65, 38, 82, 0]
    rf_results = [87, 58, 43, 66, 0]
    knn_results = [91, 56, 5, 15, 0]
    title = 'Recall of Models on Best Seeds of Magicseaweed Rating Datasets (4 locations)'
    x_labels = ['Flat', 'Poor', 'Fair', 'Good', 'Epic']
    x_axis_label = 'Category'
    y_axis_label = 'Recall'
    create_evaluation_bar_chart(cnn_results, svm_results, rf_results, knn_results, title, x_labels, x_axis_label,
                                y_axis_label)


def magicseaweed_rating_recall_5_chart():
    """CNN, SVM, RF, and KNN models were trained and tested on datasets that contained images of up to 5 surfing
    locations from Magicseaweed. Each image was classed as 'flat', 'poor', 'fair', 'good', or 'epic' depending on how
    good the conditions in the images were for surfing. This function creates a bar chart showing the recall the
    models achieved on the best seed of the dataset that contained images from all 5 surfing locations (Bantham Beach,
    Polzeath, Porthowan, Praa Sands, and Widemouth Bay).
    """
    cnn_results = [68, 78, 50, 57, 0]
    svm_results = [62, 76, 36, 71, 0]
    rf_results = [65, 78, 36, 80, 0]
    knn_results = [68, 71, 5, 17, 0]
    title = 'Recall of Models on Best Seeds of Magicseaweed Rating Datasets (5 locations)'
    x_labels = ['Flat', 'Poor', 'Fair', 'Good', 'Epic']
    x_axis_label = 'Category'
    y_axis_label = 'Recall'
    create_evaluation_bar_chart(cnn_results, svm_results, rf_results, knn_results, title, x_labels, x_axis_label,
                                y_axis_label)


def surfline_rating_accuracy_chart():
    """CNN, SVM, RF, and KNN models were trained and tested on datasets that contained images of up to 5 surfing
    locations from Surfline. Each image was classed as 'flat', 'poor', 'fair', 'good', or 'epic' depending on how
    good the conditions in the images were for surfing. This function creates a bar chart showing the accuracy the
    models achieved on the best seeds that were found for each of these datasets.
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


def surfline_rating_precision_1_chart():
    """CNN, SVM, RF, and KNN models were trained and tested on datasets that contained images of up to 5 surfing
    locations from Surfline. Each image was classed as 'flat', 'poor', 'fair', 'good', or 'epic' depending on how
    good the conditions in the images were for surfing. This function creates a bar chart showing the precision the
    models achieved on the best seed of the dataset that contained images from 1 surfing location (Pipeline).
    """
    cnn_results = [38, 72, 53, 67, 67]
    svm_results = [100, 61, 58, 57, 100]
    rf_results = [40, 70, 64, 50, 0]
    knn_results = [38, 51, 50, 100, 0]
    title = 'Precision of Models on Best Seeds of Surfline Rating Datasets (1 location)'
    x_labels = ['Flat', 'Poor', 'Fair', 'Good', 'Epic']
    x_axis_label = 'Category'
    y_axis_label = 'Precision'
    create_evaluation_bar_chart(cnn_results, svm_results, rf_results, knn_results, title, x_labels, x_axis_label,
                                y_axis_label)


def surfline_rating_precision_2_chart():
    """CNN, SVM, RF, and KNN models were trained and tested on datasets that contained images of up to 5 surfing
    locations from Surfline. Each image was classed as 'flat', 'poor', 'fair', 'good', or 'epic' depending on how
    good the conditions in the images were for surfing. This function creates a bar chart showing the precision the
    models achieved on the best seed of the dataset that contained images from 2 surfing locations (Pipeline and
    Lorne Point).
    """
    cnn_results = [71, 60, 68, 50, 100]
    svm_results = [86, 70, 50, 50, 50]
    rf_results = [62, 50, 71, 100, 0]
    knn_results = [45, 38, 57, 0, 0]
    title = 'Precision of Models on Best Seeds of Surfline Rating Datasets (2 locations)'
    x_labels = ['Flat', 'Poor', 'Fair', 'Good', 'Epic']
    x_axis_label = 'Category'
    y_axis_label = 'Precision'
    create_evaluation_bar_chart(cnn_results, svm_results, rf_results, knn_results, title, x_labels, x_axis_label,
                                y_axis_label)


def surfline_rating_precision_3_chart():
    """CNN, SVM, RF, and KNN models were trained and tested on datasets that contained images of up to 5 surfing
    locations from Surfline. Each image was classed as 'flat', 'poor', 'fair', 'good', or 'epic' depending on how
    good the conditions in the images were for surfing. This function creates a bar chart showing the precision the
    models achieved on the best seed of the dataset that contained images from 3 surfing locations (Pipeline, Lorne
    Point, and Noosa Heads).
    """
    cnn_results = [69, 65, 95, 100, 0]
    svm_results = [75, 57, 87, 50, 0]
    rf_results = [71, 65, 81, 67, 0]
    knn_results = [47, 44, 80, 33, 0]
    title = 'Precision of Models on Best Seeds of Surfline Rating Datasets (3 locations)'
    x_labels = ['Flat', 'Poor', 'Fair', 'Good', 'Epic']
    x_axis_label = 'Category'
    y_axis_label = 'Precision'
    create_evaluation_bar_chart(cnn_results, svm_results, rf_results, knn_results, title, x_labels, x_axis_label,
                                y_axis_label)


def surfline_rating_precision_4_chart():
    """CNN, SVM, RF, and KNN models were trained and tested on datasets that contained images of up to 5 surfing
    locations from Surfline. Each image was classed as 'flat', 'poor', 'fair', 'good', or 'epic' depending on how
    good the conditions in the images were for surfing. This function creates a bar chart showing the precision the
    models achieved on the best seed of the dataset that contained images from 4 surfing locations (Pipeline, Lorne
    Point, Noosa Heads, and Maria's Beachfront).
    """
    cnn_results = [72, 70, 80, 50, 0]
    svm_results = [64, 59, 81, 50, 100]
    rf_results = [70, 70, 71, 60, 0]
    knn_results = [45, 45, 62, 0, 0]
    title = 'Precision of Models on Best Seeds of Surfline Rating Datasets (4 locations)'
    x_labels = ['Flat', 'Poor', 'Fair', 'Good', 'Epic']
    x_axis_label = 'Category'
    y_axis_label = 'Precision'
    create_evaluation_bar_chart(cnn_results, svm_results, rf_results, knn_results, title, x_labels, x_axis_label,
                                y_axis_label)


def surfline_rating_precision_5_chart():
    """CNN, SVM, RF, and KNN models were trained and tested on datasets that contained images of up to 5 surfing
    locations from Surfline. Each image was classed as 'flat', 'poor', 'fair', 'good', or 'epic' depending on how
    good the conditions in the images were for surfing. This function creates a bar chart showing the precision the
    models achieved on the best seed of the dataset that contained images from all 5 surfing locations (Pipeline, Lorne
    Point, Noosa Heads, Maria's Beachfront, and Rocky Point).
    """
    cnn_results = [88, 64, 75, 75, 50]
    svm_results = [68, 67, 85, 53, 0]
    rf_results = [69, 84, 76, 65, 0]
    knn_results = [51, 35, 67, 75, 0]
    title = 'Precision of Models on Best Seeds of Surfline Rating Datasets (5 locations)'
    x_labels = ['Flat', 'Poor', 'Fair', 'Good', 'Epic']
    x_axis_label = 'Category'
    y_axis_label = 'Precision'
    create_evaluation_bar_chart(cnn_results, svm_results, rf_results, knn_results, title, x_labels, x_axis_label,
                                y_axis_label)


def surfline_rating_recall_1_chart():
    """CNN, SVM, RF, and KNN models were trained and tested on datasets that contained images of up to 5 surfing
    locations from Surfline. Each image was classed as 'flat', 'poor', 'fair', 'good', or 'epic' depending on how
    good the conditions in the images were for surfing. This function creates a bar chart showing the recall the
    models achieved on the best seed of the dataset that contained images from 1 surfing location (Pipeline).
    """
    cnn_results = [33, 69, 62, 60, 100]
    svm_results = [38, 85, 69, 44, 50]
    rf_results = [44, 81, 69, 30, 0]
    knn_results = [60, 74, 15, 14, 0]
    title = 'Recall of Models on Best Seeds of Surfline Rating Datasets (1 location)'
    x_labels = ['Flat', 'Poor', 'Fair', 'Good', 'Epic']
    x_axis_label = 'Category'
    y_axis_label = 'Recall'
    create_evaluation_bar_chart(cnn_results, svm_results, rf_results, knn_results, title, x_labels, x_axis_label,
                                y_axis_label)


def surfline_rating_recall_2_chart():
    """CNN, SVM, RF, and KNN models were trained and tested on datasets that contained images of up to 5 surfing
    locations from Surfline. Each image was classed as 'flat', 'poor', 'fair', 'good', or 'epic' depending on how
    good the conditions in the images were for surfing. This function creates a bar chart showing the recall the
    models achieved on the best seed of the dataset that contained images from 2 surfing locations (Pipeline and Lorne
    Point).
    """
    cnn_results = [81, 41, 73, 67, 50]
    svm_results = [78, 73, 76, 29, 25]
    rf_results = [71, 45, 85, 44, 0]
    knn_results = [85, 38, 24, 0, 0]
    title = 'Recall of Models on Best Seeds of Surfline Rating Datasets (2 locations)'
    x_labels = ['Flat', 'Poor', 'Fair', 'Good', 'Epic']
    x_axis_label = 'Category'
    y_axis_label = 'Recall'
    create_evaluation_bar_chart(cnn_results, svm_results, rf_results, knn_results, title, x_labels, x_axis_label,
                                y_axis_label)


def surfline_rating_recall_3_chart():
    """CNN, SVM, RF, and KNN models were trained and tested on datasets that contained images of up to 5 surfing
    locations from Surfline. Each image was classed as 'flat', 'poor', 'fair', 'good', or 'epic' depending on how
    good the conditions in the images were for surfing. This function creates a bar chart showing the recall the
    models achieved on the best seed of the dataset that contained images from 3 surfing locations (Pipeline, Lorne
    Point, and Noosa Heads).
    """
    cnn_results = [80, 54, 97, 62, 0]
    svm_results = [60, 71, 87, 50, 0]
    rf_results = [73, 62, 92, 25, 0]
    knn_results = [73, 67, 14, 12, 0]
    title = 'Recall of Models on Best Seeds of Surfline Rating Datasets (3 locations)'
    x_labels = ['Flat', 'Poor', 'Fair', 'Good', 'Epic']
    x_axis_label = 'Category'
    y_axis_label = 'Recall'
    create_evaluation_bar_chart(cnn_results, svm_results, rf_results, knn_results, title, x_labels, x_axis_label,
                                y_axis_label)


def surfline_rating_recall_4_chart():
    """CNN, SVM, RF, and KNN models were trained and tested on datasets that contained images of up to 5 surfing
    locations from Surfline. Each image was classed as 'flat', 'poor', 'fair', 'good', or 'epic' depending on how
    good the conditions in the images were for surfing. This function creates a bar chart showing the recall the
    models achieved on the best seed of the dataset that contained images from 4 surfing locations (Pipeline, Lorne
    Point, Noosa Heads, and Maria's Beachfront).
    """
    cnn_results = [82, 45, 88, 62, 0]
    svm_results = [62, 63, 86, 50, 20]
    rf_results = [78, 52, 85, 38, 0]
    knn_results = [92, 47, 13, 0, 0]
    title = 'Recall of Models on Best Seeds of Surfline Rating Datasets (4 locations)'
    x_labels = ['Flat', 'Poor', 'Fair', 'Good', 'Epic']
    x_axis_label = 'Category'
    y_axis_label = 'Recall'
    create_evaluation_bar_chart(cnn_results, svm_results, rf_results, knn_results, title, x_labels, x_axis_label,
                                y_axis_label)


def surfline_rating_recall_5_chart():
    """CNN, SVM, RF, and KNN models were trained and tested on datasets that contained images of up to 5 surfing
    locations from Surfline. Each image was classed as 'flat', 'poor', 'fair', 'good', or 'epic' depending on how
    good the conditions in the images were for surfing. This function creates a bar chart showing the recall the
    models achieved on the best seed of the dataset that contained images from all 5 surfing locations (Pipeline, Lorne
    Point, Noosa Heads, Maria's Beachfront, and Rocky Point).
    """
    cnn_results = [62, 78, 91, 38, 100]
    svm_results = [81, 56, 81, 50, 0]
    rf_results = [92, 50, 76, 69, 0]
    knn_results = [88, 57, 9, 16, 0]
    title = 'Recall of Models on Best Seeds of Surfline Rating Datasets (5 locations)'
    x_labels = ['Flat', 'Poor', 'Fair', 'Good', 'Epic']
    x_axis_label = 'Category'
    y_axis_label = 'Recall'
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

    # Precision of Models on Best Seeds of Magicseaweed Rating Datasets (1 location)
    # magicseaweed_rating_precision_1_chart()

    # Precision of Models on Best Seeds of Magicseaweed Rating Datasets (2 locations)
    # magicseaweed_rating_precision_2_chart()

    # Precision of Models on Best Seeds of Magicseaweed Rating Datasets (3 locations)
    # magicseaweed_rating_precision_3_chart()

    # Precision of Models on Best Seeds of Magicseaweed Rating Datasets (4 locations)
    # magicseaweed_rating_precision_4_chart()

    # Precision of Models on Best Seeds of Magicseaweed Rating Datasets (5 locations)
    # magicseaweed_rating_precision_5_chart()

    # Recall of Models on Best Seeds of Magicseaweed Rating Datasets (1 location)
    # magicseaweed_rating_recall_1_chart()

    # Recall of Models on Best Seeds of Magicseaweed Rating Datasets (2 locations)
    # magicseaweed_rating_recall_2_chart()

    # Recall of Models on Best Seeds of Magicseaweed Rating Datasets (3 locations)
    # magicseaweed_rating_recall_3_chart()

    # Recall of Models on Best Seeds of Magicseaweed Rating Datasets (4 locations)
    # magicseaweed_rating_recall_4_chart()

    # Recall of Models on Best Seeds of Magicseaweed Rating Datasets (5 locations)
    # magicseaweed_rating_recall_5_chart()

    # Accuracy of Models on Best Seeds of Surfline Rating Datasets
    # surfline_rating_accuracy_chart()

    # Precision of Models on Best Seeds of Surfline Rating Datasets (1 location)
    # surfline_rating_precision_1_chart()

    # Precision of Models on Best Seeds of Surfline Rating Datasets (2 locations)
    # surfline_rating_precision_2_chart()

    # Precision of Models on Best Seeds of Surfline Rating Datasets (3 locations)
    # surfline_rating_precision_3_chart()

    # Precision of Models on Best Seeds of Surfline Rating Datasets (4 locations)
    # surfline_rating_precision_4_chart()

    # Precision of Models on Best Seeds of Surfline Rating Datasets (5 locations)
    # surfline_rating_precision_5_chart()

    # Recall of Models on Best Seeds of Surfline Rating Datasets (1 location)
    # surfline_rating_recall_1_chart()

    # Recall of Models on Best Seeds of Surfline Rating Datasets (2 locations)
    # surfline_rating_recall_2_chart()

    # Recall of Models on Best Seeds of Surfline Rating Datasets (3 locations)
    # surfline_rating_recall_3_chart()

    # Recall of Models on Best Seeds of Surfline Rating Datasets (4 locations)
    # surfline_rating_recall_4_chart()

    # Recall of Models on Best Seeds of Surfline Rating Datasets (5 locations)
    # surfline_rating_recall_5_chart()


if __name__ == '__main__':
    main()
