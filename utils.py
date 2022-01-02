from tensorflow import keras
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import os


def extract_images_and_labels(path_to_dir, categories, color_mode, image_size):
    """Extracts images and labels from a directory.

    Args:
        path_to_dir (str): The path to the directory containing the folders with the images.
        categories (list[str]): An array containing the names of the folders in the directory that have the images. The
            index of each item in the array represents the label that category will be given.
        color_mode (str): Sets the color mode of the images in the folders. Either 'grayscale', 'rgb', or 'rgba'.
        image_size (int, int): All of the images in the folders will be converted to this size.

    Returns:
        - X (list[list]): The images that were extracted.
        - y (list[int]): The labels of the images.
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
        int: Number of channels.
    """
    if color_mode == 'grayscale':
        return 1
    elif color_mode == 'rgb':
        return 3
    else:
        return 4


def plot_confusion_matrix(y_test, y_predicted):
    """Plots a confusion matrix showing the model's predictions versus what the correct answers were.

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


def plot_cnn_loss(history, epochs):
    """Plots a graph showing the training and validation loss over each epoch.

    Args:
        history (): The training metrics for each epoch.
        epochs (int): The number of epochs used to train the model.
    """
    epochs_range = range(1, epochs + 1)
    plt.plot(epochs_range, history.history['loss'], 'green', label='Training loss')
    plt.plot(epochs_range, history.history['val_loss'], 'blue', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def plot_cnn_accuracy(history, epochs):
    """Plots a graph showing the training and validation accuracy over each epoch.

    Args:
        history (): The training metrics for each epoch.
        epochs (int): The number of epochs used to train the model.
    """
    epochs_range = range(1, epochs + 1)
    plt.plot(epochs_range, history.history['accuracy'], 'green', label='Training accuracy')
    plt.plot(epochs_range, history.history['val_accuracy'], 'blue', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def cnn_model_accuracy_and_loss(model, X_val, y_val, X_test, y_test):
    """Prints the accuracy and loss of a CNN model on a validation dataset and a test dataset.

    Args:
        model (): The CNN model whose details will be displayed.
        X_val (ndarray): The validation dataset images.
        y_val (ndarray): The validation dataset labels.
        X_test (ndarray): The test dataset images.
        y_test (ndarray): The test dataset labels.
    """
    loss, acc = model.evaluate(X_val, y_val, verbose=0)
    print('Validation dataset accuracy: ' + str(round(acc * 100, 2)) + '%')
    print('Validation dataset loss: ' + str(round(loss, 6)))
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print('Test dataset accuracy: ' + str(round(acc * 100, 2)) + '%')
    print('Test dataset loss: ' + str(round(loss, 6)))


def model_test_accuracy(model, X_test, y_test):
    """Prints the accuracy of a model on a test dataset. It is used by the sklearn models.

    Args:
        model (): The model whose details will be displayed.
        X_test (list[list]): The test dataset images.
        y_test (list[int]): The test dataset labels.
    """
    accuracy = model.score(X_test, y_test)
    print('Test Dataset Accuracy: ' + str(accuracy))
    print('Number of correct predictions: {}/{}'.format(round(len(X_test) * accuracy), len(X_test)))
