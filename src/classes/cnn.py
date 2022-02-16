from utils import *
from tensorflow import keras
from keras import layers
from keras.callbacks import History
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
import matplotlib.pyplot as plt
import numpy as np


class BaseCNN:
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

        # Create the CNN model
        self.model, self.model_name = self.create_model()

        # Compile the model
        self.optimizer = keras.optimizers.Adam()
        self.loss = keras.losses.BinaryCrossentropy(from_logits=True)
        self.metrics = ['accuracy']
        self.compile_model()

    def print_model_name(self):
        """Prints the name of the current model."""
        print("\n" + self.model_name)

    def create_model(self):
        """Creates the model.

        Raises:
            NotImplementedError: Subclasses must implement this method.

        """
        raise NotImplementedError

    def compile_model(self):
        """Compiles the model."""
        self.model.compile(optimizer=self.optimizer,loss=self.loss, metrics=self.metrics)

    def train_model(self, X_train, X_val, y_train, y_val, save_name=None):
        """Trains the model using training and validation data. The best model during training can optionally be saved.

        Args:
            X_train (np.ndarray): The training dataset images. Each image is represented as a pixel array.
            X_val (np.ndarray): The validation dataset images. Each image is represented as a pixel array.
            y_train (np.ndarray): The training dataset labels.
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
        self.print_model_name()
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

    def summary(self):
        """Prints information about the CNN model and its structure."""
        self.model.summary()

    def kfold_cross_validation(self, X, y, n_splits):
        """K-Fold Cross Validation is applied to the model and info about accuracy, precision, and recall is printed.

        Note: This function will only return legitimate results if the model has not been trained on the dataset that is
        used.

        Args:
            X (np.ndarray): The images in the dataset, where each image is represented as a list of pixel values.
            y (np.ndarray): The labels of the images.
            n_splits (int): The number of folds the dataset will be divided into.

        """
        fold_num = 1
        X = np.array(X)
        y = np.array(y)
        accuracy_scores = []
        skf = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2)

        for train_index, test_index in skf.split(X, y):
            # Set up training, validation, and test datasets for the current fold
            X = X.reshape(len(X), self.image_height, self.image_width, self.num_channels)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

            # Create a clone of the current model and compile it
            cloned_model = keras.models.clone_model(self.model)
            cloned_model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

            # Train the cloned model using the training and validation data
            print(f'Training on fold {fold_num}')
            cloned_model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=self.epochs,
                batch_size=self.batch_size
            )

            # Test the cloned model on the test data, then print the results
            loss, acc = cloned_model.evaluate(X_test, y_test, verbose=0)
            accuracy_scores.append(acc)
            print(f'\nFold {fold_num} test dataset results:')
            print(f'Accuracy: {round(acc * 100, 2)}%')
            print(f'Loss: {loss}')
            fold_num += 1

        # Print overall k-fold cross validation results
        print(f'\n{n_splits}-Fold Cross Validation: {self.model_name}')
        print('Max accuracy: {}% '.format(round(np.max(accuracy_scores) * 100, 2)))
        print('Min accuracy: {}% '.format(round(np.min(accuracy_scores) * 100, 2)))
        print('Average accuracy: {}% '.format(round(np.average(accuracy_scores) * 100, 2)))


class LoadedCNN(BaseCNN):
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
        return NotImplementedError

    def load_model(self, model_name):
        """Loads a CNN model from the 'saved_models' folder.

        Args:
            model_name (str): The name of the model to be loaded, e.g. 'basic_cnn'.

        """
        self.model = keras.models.load_model('../saved_models/{}.h5'.format(model_name))
        self.model_name = model_name


class CNN(BaseCNN):
    """An implementation of a Keras CNN model."""

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
        model_name = 'cnn_model'
        data_augmentation = keras.Sequential([
            layers.RandomFlip("horizontal", input_shape=(self.image_height, self.image_width, self.num_channels)),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ])
        model = keras.Sequential([
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
        return model, model_name
