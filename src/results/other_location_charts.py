"""This module creates charts showing how the models that were trained on images of one surfing location performed when
tested on images of other surfing locations. """

from utils.helper_utils import create_evaluation_bar_chart_1


def binary_bantham_other_locations_accuracy_chart():
    """CNN, SVM, RF, and KNN models were trained using the dataset containing images of the Bantham Beach. They were
    then tested on each of the datasets containing images of the other Magicseaweed surfing locations. This function
    creates a chart showing the accuracy the models achieved when tested on the datasets of the other Magicseaweed
    surfing locations.
    """
    cnn_results = [94, 87, 95, 95]
    svm_results = [79, 73, 87, 84]
    rf_results = [67, 80, 73, 74]
    knn_results = [57, 63, 70, 70]
    title = 'Accuracy of Binary Bantham Models on Other Magicseaweed Surfing Locations'
    x_labels = ['Polzeath', 'Porthowan', 'Praa Sands', 'Widemouth Bay']
    x_axis_label = 'Surfing location'
    y_axis_label = 'Accuracy (%)'
    create_evaluation_bar_chart_1(cnn_results, svm_results, rf_results, knn_results, title, x_labels, x_axis_label,
                                  y_axis_label)


def binary_bantham_other_locations_precision_chart():
    """CNN, SVM, RF, and KNN models were trained using the dataset containing images of the Bantham Beach. They were
    then tested on each of the datasets containing images of the other Magicseaweed surfing locations. This function
    creates a chart showing the precision the models achieved when tested on the datasets of the other Magicseaweed
    surfing locations.
    """
    cnn_results = [89, 81, 92, 92]
    svm_results = [73, 67, 80, 77]
    rf_results = [64, 73, 66, 68]
    knn_results = [67, 72, 88, 92]
    title = 'Precision of Binary Bantham Models on Other Magicseaweed Surfing Locations'
    x_labels = ['Polzeath', 'Porthowan', 'Praa Sands', 'Widemouth Bay']
    x_axis_label = 'Surfing location'
    y_axis_label = 'Precision'
    create_evaluation_bar_chart_1(cnn_results, svm_results, rf_results, knn_results, title, x_labels, x_axis_label,
                                  y_axis_label)


def binary_bantham_other_locations_recall_chart():
    """CNN, SVM, RF, and KNN models were trained using the dataset containing images of the Bantham Beach. They were
    then tested on each of the datasets containing images of the other Magicseaweed surfing locations. This function
    creates a chart showing the recall the models achieved when tested on the datasets of the other Magicseaweed
    surfing locations.
    """
    cnn_results = [100, 96, 98, 98]
    svm_results = [92, 92, 98, 98]
    rf_results = [76, 96, 96, 92]
    knn_results = [28, 42, 46, 44]
    title = 'Recall of Binary Bantham Models on Other Magicseaweed Surfing Locations'
    x_labels = ['Polzeath', 'Porthowan', 'Praa Sands', 'Widemouth Bay']
    x_axis_label = 'Surfing location'
    y_axis_label = 'Recall'
    create_evaluation_bar_chart_1(cnn_results, svm_results, rf_results, knn_results, title, x_labels, x_axis_label,
                                  y_axis_label)


def binary_pipeline_other_locations_accuracy_chart():
    """CNN, SVM, RF, and KNN models were trained using the dataset containing images of Pipeline. They were
    then tested on each of the datasets containing images of the other Surfline surfing locations. This function
    creates a chart showing the accuracy the models achieved when tested on the datasets of the other Surfline
    surfing locations.
    """
    cnn_results = [80, 94, 98, 77]
    svm_results = [74, 78, 78, 63]
    rf_results = [66, 59, 72, 53]
    knn_results = [63, 51, 51, 48]
    title = 'Accuracy of Binary Pipeline Models on Other Surfline Surfing Locations'
    x_labels = ['Lorne Point', 'Noosa Heads', 'Maria\'s Beachfront', 'Rocky Point']
    x_axis_label = 'Surfing location'
    y_axis_label = 'Accuracy (%)'
    create_evaluation_bar_chart_1(cnn_results, svm_results, rf_results, knn_results, title, x_labels, x_axis_label,
                                  y_axis_label)


def binary_pipeline_other_locations_precision_chart():
    """CNN, SVM, RF, and KNN models were trained using the dataset containing images of Pipeline. They were
    then tested on each of the datasets containing images of the other Surfline surfing locations. This function
    creates a chart showing the precision the models achieved when tested on the datasets of the other Surfline
    surfing locations.
    """
    cnn_results = [100, 100, 96, 100]
    svm_results = [100, 100, 97, 58]
    rf_results = [100, 91, 96, 58]
    knn_results = [100, 67, 100, 45]
    title = 'Precision of Binary Pipeline Models on Other Surfline Surfing Locations'
    x_labels = ['Lorne Point', 'Noosa Heads', 'Maria\'s Beachfront', 'Rocky Point']
    x_axis_label = 'Surfing location'
    y_axis_label = 'Precision'
    create_evaluation_bar_chart_1(cnn_results, svm_results, rf_results, knn_results, title, x_labels, x_axis_label,
                                  y_axis_label)

def binary_pipeline_other_locations_recall_chart():
    """CNN, SVM, RF, and KNN models were trained using the dataset containing images of Pipeline. They were
    then tested on each of the datasets containing images of the other Surfline surfing locations. This function
    creates a chart showing the recall the models achieved when tested on the datasets of the other Surfline
    surfing locations.
    """
    cnn_results = [60, 88, 100, 54]
    svm_results = [48, 56, 58, 90]
    rf_results = [32, 20, 46, 22]
    knn_results = [26, 4, 2, 18]
    title = 'Recall of Binary Pipeline Models on Other Surfline Surfing Locations'
    x_labels = ['Lorne Point', 'Noosa Heads', 'Maria\'s Beachfront', 'Rocky Point']
    x_axis_label = 'Surfing location'
    y_axis_label = 'Recall'
    create_evaluation_bar_chart_1(cnn_results, svm_results, rf_results, knn_results, title, x_labels, x_axis_label,
                                  y_axis_label)


def main():
    # Accuracy of Binary Bantham Models on Other Magicseaweed Surfing Locations
    binary_bantham_other_locations_accuracy_chart()

    # Precision of Binary Bantham Models on Other Magicseaweed Surfing Locations
    binary_bantham_other_locations_precision_chart()

    # Recall of Binary Bantham Models on Other Magicseaweed Surfing Locations
    binary_bantham_other_locations_recall_chart()

    # Accuracy of Binary Pipeline Models on Other Surfline Surfing Locations
    binary_pipeline_other_locations_accuracy_chart()

    # Precision of Binary Pipeline Models on Other Surfline Surfing Locations
    binary_pipeline_other_locations_precision_chart()

    # Recall of Binary Pipeline Models on Other Surfline Surfing Locations
    binary_pipeline_other_locations_recall_chart()


if __name__ == '__main__':
    main()
