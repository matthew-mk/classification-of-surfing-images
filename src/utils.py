from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, cross_validate
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import os


def extract_images_and_labels(path_to_dir, categories, color_mode, image_size):
    """Extracts datasets and labels from a directory that has folders with datasets for different classes.

    The names of the folders that are specified in the categories array are looped through, with each folder
    containing the datasets for a particular class. The datasets in each folder are assigned a label; the index of the
    name of the folder in the categories array. Each of the datasets in the folders are converted to be the size and
    color mode that are specified as parameters. Each image is transformed into a numpy array containing the image's
    pixel values and the pixel values are normalized.

    Args:
        path_to_dir (str): The path to the directory containing the folders with the datasets.
        categories (list[str]): An array containing the names of the folders in the directory that have the datasets. The
            index of each item in the array represents the label that category will be given.
        color_mode (str): Sets the color mode of the datasets in the folders. Either 'grayscale', 'rgb', or 'rgba'.
        image_size ((int, int)): All of the datasets in the folders will be converted to this size.

    Returns:
        (tuple): tuple containing:

            - X (list[list]): The datasets that were extracted.
            - y (list[int]): The labels of the datasets.
    """
    X = []
    y = []
    for category in categories:
        category_path = os.path.join(path_to_dir, category)
        category_label = categories.index(category)
        for file in os.listdir(category_path):
            if file.endswith('.png'):
                img_path = os.path.join(category_path, file)
                img = keras.preprocessing.image.load_img(img_path,
                                                         color_mode=color_mode,
                                                         target_size=image_size)
                img_array = np.array(img).flatten()
                img_array = img_array / 255  # normalize the data
                X.append(img_array)
                y.append(category_label)
    return X, y


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


def plot_confusion_matrix(y_test, y_predicted):
    """Creates a confusion matrix showing the model's predictions versus what the correct answers were.

    Args:
        y_test (list[int]): The correct answers.
        y_predicted (list[int]): The predictions that the model made.

    """
    cm = confusion_matrix(y_test, y_predicted)
    plt.figure(figsize=(4, 4))
    sn.heatmap(cm, annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.show()


def test_model(model, is_cnn, X_test, y_test):
    """Tests the model on a dataset and prints the accuracy and loss.

    Args:
        model (keras.Model): The model that will be tested.
        is_cnn (bool): True if model is a CNN model, otherwise False.
        X_test (np.ndarray): The datasets in the dataset. Each image is represented as a pixel array.
        y_test (np.ndarray): The labels in the dataset.

    """
    if is_cnn:
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        print('Accuracy: ' + str(round(acc * 100, 2)) + '%')
        print('Number of correct predictions: {}/{}'.format(round(len(X_test) * acc), len(X_test)))
        print('Loss: ' + str(round(loss, 6)))
    else:
        acc = model.score(X_test, y_test)
        print('Accuracy: {}%'.format(round(acc * 100, 2)))
        print('Number of correct predictions: {}/{}'.format(round(len(X_test) * acc), len(X_test)))


def kfold_cross_validation(models, X, y, folds):
    """K-Fold Cross Validation is applied to saved_models and details are printed.

    K-Fold Cross Validation is applied to saved_models and details about the accuracy, precision, recall, and f1 score are
    printed.

    Args:
        models (list[(str, keras.Model)]): A list of tuples, with each tuple containing the name of a model, along with
            the model itself.
        X (list[list]): The datasets in the dataset. Each image is represented as a list of pixel values.
        y (list[int]): The labels of the datasets.
        folds (int): The number of folds the dataset will be divided into.

    """
    scoring_types = ['accuracy', 'precision', 'recall', 'f1']
    for name, model in models:
        print('\n{} {}-Fold Cross Validation'.format(name, folds))
        cv_results = cross_validate(model, X, y, cv=folds, scoring=scoring_types)
        for scoring_type in scoring_types:
            cv_result = cv_results['test_{}'.format(scoring_type)]
            print('Max {}: {}% '.format(scoring_type, round(np.max(cv_result) * 100, 2)))
            print('Min {}: {}% '.format(scoring_type, round(np.min(cv_result) * 100, 2)))
            print('Average {}: {}% '.format(scoring_type, round(np.mean(cv_result) * 100, 2)))
