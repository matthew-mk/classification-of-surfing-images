from utils import *
from tensorflow import keras
from keras import layers
from keras.callbacks import History
import matplotlib.pyplot as plt
import numpy as np


class CNN:
    """The base class of a CNN model.


    """
    def __init__(self, config):
        """The constructor for a CNN model.

        Args:
            config ():

        """
        self.image_height = config['image_height']
        self.image_width = config['image_width']
        self.image_size = (self.image_height, self.image_width)
        self.color_mode = config['color_mode']
        self.num_channels = get_channels(self.color_mode)
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.history = History()
        self.model = None
        self.model_name = 'unnamed_cnn'
        self.create_model()

    def create_model(self):
        """Creates the architecture of the CNN model.

        Raises:
            NotImplementedError: Subclasses must implement this method.

        """
        raise NotImplementedError

    def get_model(self):
        """Returns the CNN model.

        Returns:
            model (keras.Model): The CNN model.

        """
        return self.model

    def get_model_name(self):
        """Returns the name of the current model.

        Returns:
            The name of the model.

        """
        return self.model_name

    def compile_model(self, optimizer, loss, metrics):
        """Compiles the CNN model using the specified optimizer, loss function, and metrics.

        Args:
            optimizer (keras.optimizers.Optimizer): The optimizer.
            loss (keras.losses.Loss): The loss function.
            metrics (list[keras.metrics.Metric]): A list of metrics that the model will evaluate during training and
                testing.

        """
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train_model(self, X_train, y_train, X_val, y_val, save_name=None):
        """Trains the model using training and validation data. The best model during training can optionally be saved.

        Args:
            X_train (np.ndarray): The training dataset images. Each image is represented as a pixel array.
            y_train (np.ndarray): The training dataset labels.
            X_val (np.ndarray): The validation dataset images. Each image is represented as a pixel array.
            y_val (np.ndarray): The validation dataset labels.
            save_name (str): Optional parameter. If included, the best model during training will be saved using this
                name. Defaults to None.

        """
        if save_name is None:
            callbacks = []
        else:
            model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
                '../saved_models/{}.h5'.format(save_name),
                monitor='val_loss',
                mode='min',
                save_best_only=True
            )
            callbacks = [model_checkpoint_callback]
            self.model_name = save_name

        self.history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks
        )

    def plot_training_accuracy(self):
        """Plots a graph showing the CNN model's training and validation accuracy over each epoch."""
        epochs_range = range(1, self.epochs + 1)
        plt.plot(epochs_range, self.history.history['accuracy'], label='Training accuracy')
        plt.plot(epochs_range, self.history.history['val_accuracy'], label='Validation accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

    def plot_training_loss(self):
        """Plots a graph showing the CNN model's training and validation loss over each epoch."""
        epochs_range = range(1, self.epochs + 1)
        plt.plot(epochs_range, self.history.history['loss'], label='Training loss')
        plt.plot(epochs_range, self.history.history['val_loss'], label='Validation loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def test_model(self, X_test, y_test):
        """Tests the model on a dataset and prints the accuracy, number of correct predictions, and loss.

        Args:
            X_test (np.ndarray): The images in the dataset, where each image is represented as a pixel array.
            y_test (np.ndarray): The labels in the dataset.

        """
        loss, acc = self.model.evaluate(X_test, y_test, verbose=0)
        print('Accuracy: {}%'.format((round(acc * 100, 2))))
        print('Number of correct predictions: {}/{}'.format(round(len(X_test) * acc), len(X_test)))
        print('Loss: {}'.format(round(loss, 6)))

    def save_model(self, model_name):
        """Saves the current CNN model in the 'saved_models' folder.

        Args:
            model_name (str): The name the model will be saved as, e.g. 'basic_cnn'.

        """
        self.model.save('../saved_models/{}.h5'.format(model_name))
        self.model_name = model_name

    def load_model(self, model_name):
        """Loads a CNN model from the 'saved_models' folder and sets it to be the model.

        Args:
            model_name (str): The name of the model to be loaded, e.g. 'basic_cnn'.

        """
        self.model = keras.models.load_model('../saved_models/{}.h5'.format(model_name))
        self.model_name = model_name

    def model_summary(self):
        """Prints information about the CNN model and its structure."""
        self.model.summary()

    def kfold_cross_validation(self, X, y, folds):
        """K-Fold Cross Validation is applied to the model and info about accuracy, precision, and recall is printed.

        Note: This function will only return legitimate results if the model has not been trained on the dataset that is
        used.

        Args:
            X (np.ndarray): The images in the dataset, where each image is represented as a list of pixel values.
            y (np.ndarray): The labels of the images.
            folds (int): The number of folds the dataset will be divided into.

        """


class LoadedCNN(CNN):
    """

    """
    def __init__(self, config, model_name):
        """

        Args:
            config ():
            model_name (str): The name of the model to be loaded.

        """
        self.image_height = config['image_height']
        self.image_width = config['image_width']
        self.image_size = (self.image_height, self.image_width)
        self.color_mode = config['color_mode']
        self.num_channels = get_channels(self.color_mode)
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.history = History()
        self.model = None
        self.model_name = model_name
        self.create_model()

    def create_model(self):
        """Creates the architecture of the CNN model."""
        self.load_model(self.model_name)
        return self.model


class BasicCNN(CNN):
    """

    """
    def __init__(self, config):
        """

        """
        super().__init__(config)
        self.compile_model(keras.optimizers.Adam(), keras.losses.BinaryCrossentropy(from_logits=True), ['accuracy'])

    def create_model(self):
        """Creates the architecture of the CNN model."""
        data_augmentation = keras.Sequential([
            layers.RandomFlip("horizontal", input_shape=(self.image_height, self.image_width, self.num_channels)),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ])
        self.model = keras.Sequential([
            data_augmentation,
            layers.Input((self.image_height, self.image_width, self.num_channels)),
            layers.Conv2D(16, 3, padding='same'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same'),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, padding='same'),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, padding='same'),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(128),
            layers.Dense(64),
            layers.Dense(1)
        ])
