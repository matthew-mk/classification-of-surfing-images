"""This module defines an abstract base class that contains common functionality for CNN models. There are also
subclasses that inherit from the base class, which are particular implementations of a CNN. """

from utils.helper_utils import *
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.callbacks import History
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np


class AbstractCNN(ABC):
    """The base class for creating, training, and evaluating a Keras CNN model."""

    def __init__(self, config):
        """Initialises the CNN model.

        Args:
            config (dict): Contains information about the images, including the image height, image width, color mode,
                and number of channels. It also includes variables that are used for training, such as the batch size
                and number of epochs.

        """
        # Init information about the images
        self.image_height = config['image_height']
        self.image_width = config['image_width']
        self.image_size = (self.image_height, self.image_width)
        self.color_mode = config['color_mode']
        self.num_channels = get_channels(self.color_mode)

        # Init information for training
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.history = History()

        # Init compilation settings
        self.optimizer = None
        self.loss = None
        self.metrics = ['accuracy']

        # Create the CNN model
        self.model, self.model_name = self.create_model()

    def print_model_name(self):
        """Prints the name of the model."""
        print("\n" + self.model_name)

    def print_classification_report(self, X_test, y_test):
        """Tests the model on a dataset and prints the precision and recall for each category.

        Args:
            X_test (np.ndarray): The images in the dataset, where each image is represented as a pixel array.
            y_test (np.ndarray): The labels in the dataset.

        """
        last_layer_index = len(self.model.layers) - 1
        last_layer_units = self.model.get_layer(index=last_layer_index).units

        if last_layer_units < 1:
            print('Classification report could not be created. The final layer in the CNN must contain at least 1 unit.')
            return
        elif last_layer_units == 1:
            y_pred = (self.model.predict(X_test) > 0.5).astype("int32")
        else:
            y_probabilities = self.model.predict(X_test)
            y_pred = tf.argmax(y_probabilities, axis=-1)

        print('Classification report:')
        print(classification_report(y_test, y_pred))

    def summary(self):
        """Prints information about the architecture of the model."""
        self.model.summary()

    @abstractmethod
    def create_model(self):
        """Creates the model.

        Raises:
            NotImplementedError: Subclasses must implement this method.

        """
        raise NotImplementedError('Subclasses must implement this method.')

    def compile_model(self, optimizer, loss):
        """Compiles the model using a particular optimizer and loss function.

        Args:
            loss (keras.losses): The loss function.
            optimizer (keras.optimizers): The optimizer, e.g. keras.optimizers.Adam().

        """
        self.optimizer = optimizer
        self.loss = loss
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

    def train_model(self, X_train, X_val, y_train, y_val, save_name=None):
        """Trains the model using training and validation data. The best model during training can optionally be saved.

        Args:
            X_train (np.ndarray): The training dataset images. Each image is represented as a pixel array.
            X_val (np.ndarray): The validation dataset images. Each image is represented as a pixel array.
            y_train (np.ndarray): The training dataset labels.
            y_val (np.ndarray): The validation dataset labels.
            save_name (str): Optional parameter, defaults to None. If included, the best model during training will be
                saved using this name. The best model is classed as the model with the smallest validation loss.

        """
        callbacks = []

        if isinstance(save_name, str) and len(save_name) > 0 and str.isspace(save_name) is False:
            # Settings to save the model
            model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
                '../../saved_models/{}.h5'.format(save_name),
                monitor='val_loss',
                mode='min',
                save_best_only=True
            )
            callbacks.append(model_checkpoint_callback)
            self.model_name = save_name

        # Train the model
        try:
            self.history = self.model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=callbacks
            )
        except RuntimeError:
            print('The model must be compiled before training/testing. Use compile_model(optimizer, loss)')

    def plot_training_accuracy(self):
        """Plots a graph showing the CNN model's training and validation accuracy over each epoch."""
        try:
            epochs_range = range(1, self.epochs + 1)
            plt.plot(epochs_range, self.history.history['accuracy'], label='Training accuracy')
            plt.plot(epochs_range, self.history.history['val_accuracy'], label='Validation accuracy')
            plt.title('Training and Validation Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.show()
        except KeyError:
            raise

    def plot_training_loss(self):
        """Plots a graph showing the CNN model's training and validation loss over each epoch."""
        try:
            epochs_range = range(1, self.epochs + 1)
            plt.plot(epochs_range, self.history.history['loss'], label='Training loss')
            plt.plot(epochs_range, self.history.history['val_loss'], label='Validation loss')
            plt.title('Training and Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()
        except KeyError:
            raise

    def plot_confusion_matrix(self, y_test, y_pred):
        """Creates a confusion matrix showing the model's predictions versus what the correct answers were.

        Args:
            y_test (list[int]): The labels of the images in the dataset.
            y_pred (list[int]): The predictions that the model made.

        """
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(4, 4))
        sn.heatmap(cm, annot=True)
        plt.xlabel('Predicted')
        plt.ylabel('Truth')
        plt.show()

    def test_model(self, X_test, y_test):
        """Tests the model on a dataset and prints the accuracy, number of correct predictions, and loss.

        Args:
            X_test (np.ndarray): The images in the dataset, where each image is represented as a pixel array.
            y_test (np.ndarray): The labels in the dataset.

        """
        try:
            loss, acc = self.model.evaluate(X_test, y_test, verbose=0)
        except RuntimeError:
            print('The model must be compiled before training/testing. Use compile_model(optimizer, loss)')

        self.print_model_name()
        print(f'Loss: {round(loss, 6)}')
        print(f'Accuracy: {round(acc * 100, 2)}%')
        print(f'Number of correct predictions: {round(len(X_test) * acc)}/{len(X_test)}')
        self.print_classification_report(X_test, y_test)

    def save_model(self, model_name):
        """Saves the current CNN model in the 'saved_models' folder.

        Args:
            model_name (str): The name the model will be saved as, e.g. 'basic_cnn'.

        """
        if isinstance(model_name, str) and len(model_name) > 0 and str.isspace(model_name) is False:
            self.model.save('../../saved_models/{}.h5'.format(model_name))
            self.model_name = model_name
        else:
            print('The model could not be saved. An invalid name was used.')

    def kfold_cross_validation(self, X, y, n_splits, test_size):
        """K-Fold Cross Validation is applied to the model and info about accuracy, precision, and recall is printed.

        Note: This function will only return legitimate results if the model has not been trained on the dataset that is
        used.

        Args:
            X (np.ndarray): The images in the dataset, where each image is represented as a list of pixel values.
            y (np.ndarray): The labels of the images.
            n_splits (int): The number of folds the dataset will be divided into.
            test_size (float): The proportion of images that will be used for testing in each fold. Ranges from 0-1.
                E.g. 0.2 means 20% of images will be used for testing.

        """
        skf = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size)
        X = np.array(X)
        y = np.array(y)
        default_weights = self.model.get_weights()
        fold_num = 1

        for train_index, test_index in skf.split(X, y):
            # Set up training, validation, and test datasets for the current fold
            X = X.reshape(len(X), self.image_height, self.image_width, self.num_channels)
            X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

            # Train and test the model on the current fold
            print(f'\nTraining on fold {fold_num}...')
            self.model.set_weights(default_weights)
            self.train_model(X_train, X_val, y_train, y_val)
            print(f'\nFold {fold_num} test dataset results:')
            self.test_model(X_test, y_test)
            fold_num += 1


class LoadedCNN(AbstractCNN):
    """A loaded Keras CNN model."""

    def __init__(self, config):
        """Initializes the information needed to train the CNN model."""
        # Init information about the images
        self.image_height = config['image_height']
        self.image_width = config['image_width']
        self.image_size = (self.image_height, self.image_width)
        self.color_mode = config['color_mode']
        self.num_channels = get_channels(self.color_mode)

        # Init information for training
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.history = History()

    def create_model(self):
        """Creates the CNN model.

        Raises:
            NotImplementedError: The model must be loaded.

        """
        raise NotImplementedError('The model must be loaded.')

    def load_model(self, model_name):
        """Loads a CNN model from the 'saved_models' folder.

        Args:
            model_name (str): The name of the model to be loaded, e.g. 'basic_cnn'.

        """
        try:
            self.model = keras.models.load_model('../../saved_models/{}.h5'.format(model_name))
            self.model_name = model_name
        except TypeError as e:
            print('Model could not be loaded')
            print(e)
        except OSError as e:
            print('Model could not be loaded')
            print(e)

class BinaryCNN(AbstractCNN):
    """A CNN model that is used for binary classification of surfing images."""

    def __init__(self, config):
        """Initializes the CNN model."""
        super().__init__(config)

    def create_model(self):
        """Creates the CNN model.

        Returns:
            tuple containing:
                - model (keras.Model): The CNN model.
                - model_name (str): The name of the model.

        """
        model_name = 'binary_cnn'
        data_augmentation = keras.Sequential([
            layers.RandomFlip("horizontal", input_shape=(self.image_height, self.image_width, self.num_channels)),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ])
        model = keras.Sequential([
            data_augmentation,
            layers.Input((self.image_height, self.image_width, self.num_channels)),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(2, activation='softmax')
        ])
        return model, model_name

class RatingCNN(AbstractCNN):
    """A CNN model that is used for multiclass classification of surfing images."""

    def __init__(self, config):
        """Initializes the CNN model."""
        super().__init__(config)

    def create_model(self):
        """Creates the CNN model.

        Returns:
            tuple containing:
                - model (keras.Model): The CNN model.
                - model_name (str): The name of the model.

        """
        model_name = 'rating_cnn'
        data_augmentation = keras.Sequential([
            layers.RandomFlip("horizontal", input_shape=(self.image_height, self.image_width, self.num_channels)),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ])
        model = keras.Sequential([
            data_augmentation,
            layers.Input((self.image_height, self.image_width, self.num_channels)),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(5, activation='softmax')
        ])
        return model, model_name
