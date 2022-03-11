"""This module defines an abstract base class that contains common functionality for Scikit-learn models. There are also
subclasses that inherit from the base class, including particular implementations of SVM, RF, and KNN models. """
import copy

from sklearn.base import clone
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import pickle


class AbstractSklearn(ABC):
    """The base class for creating, training, and evaluating a Scikit-learn model."""

    def __init__(self):
        """Initializes a Scikit-learn model."""
        self.model, self.model_name = self.create_model()

    def print_model_name(self):
        """Prints the name of the model."""
        print("\n" + self.model_name)

    @abstractmethod
    def create_model(self):
        """Creates the Scikit-learn model.

        Raises:
            NotImplementedError: Subclasses must implement this method.

        """
        raise NotImplementedError('Subclasses must implement this method.')

    def train_model(self, X_train, y_train):
        """Trains the model using a training dataset.

        Args:
            X_train (list[list]): The training dataset images, where each image is represented as a pixel array.
            y_train (list[int]): The labels of the images in the dataset.

        """
        self.model.fit(X_train, y_train)

    def test_model(self, X_test, y_test, return_acc=False):
        """Tests the model on a dataset and prints the accuracy.

        Args:
            X_test (list[list]): The images in the dataset, where each image is represented as a pixel array.
            y_test (list[int]): The labels of the images in the dataset.
            return_acc (bool): Optional variable, defaults to False. If set to true, it will return the accuracy that
                the model scored on the provided dataset.

        """
        acc = self.model.score(X_test, y_test)
        y_predicted = self.model.predict(X_test)
        self.print_model_name()
        print('Accuracy: {}%'.format(round(acc * 100, 2)))
        print('Number of correct predictions: {}/{}'.format(round(len(X_test) * acc), len(X_test)))
        print("Classification report: ")
        print(classification_report(y_test, y_predicted))
        if return_acc:
            return round(acc * 100, 2)

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

    def save_model(self, model_name):
        """Saves the current model in the 'saved_models' folder.

        Args:
            model_name (str): The name the model will be saved as, e.g. 'basic_svm'.

        """
        if isinstance(model_name, str) and len(model_name) > 0 and str.isspace(model_name) is False:
            pickle.dump(self.model, open('../../saved_models/{}.sav'.format(model_name), 'wb'))
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
        fold_num = 1

        for train_index, test_index in skf.split(X, y):
            # Set up the training and test datasets for the current fold
            X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]

            # Train and test the model on the current fold
            print(f'\nTraining on fold {fold_num}...')
            cloned_model = copy.deepcopy(self)
            cloned_model.train_model(X_train, y_train)
            print(f'Fold {fold_num} test dataset results:')
            cloned_model.test_model(X_test, y_test)
            fold_num += 1


class LoadedSklearn(AbstractSklearn):
    """A loaded Scikit-learn model."""

    def __init__(self):
        """Initializes a Scikit-learn model."""
        pass

    def create_model(self):
        """Creates the Scikit-learn model.

        Raises:
            NotImplementedError: The model must be loaded.

        """
        raise NotImplementedError('The model must be loaded.')

    def load_model(self, model_name):
        """Loads a Scikit-learn model from the 'saved_models' folder.

        Args:
            model_name (str): The name of the model to be loaded, e.g. 'basic_svm'.

        """
        try:
            self.model = pickle.load(open('../../saved_models/{}.sav'.format(model_name), 'rb'))
            self.model_name = model_name
        except TypeError as e:
            print('Model could not be loaded')
            print(e)
        except OSError as e:
            print('Model could not be loaded')
            print(e)


class SVM(AbstractSklearn):
    """An implementation of a Scikit-learn Support Vector Machine model."""

    def __init__(self):
        """Initializes a Scikit-learn model."""
        super().__init__()

    def create_model(self):
        """Creates the Scikit-learn model.

        Returns:
            tuple containing:
                - model: The Scikit-learn model.
                - model_name (str): The name of the model.

        """
        model = SVC(C=4)
        model_name = 'svm_model'
        return model, model_name

class RF(AbstractSklearn):
    """An implementation of a Scikit-learn Random Forest model."""

    def __init__(self):
        """Initializes a Scikit-learn model."""
        super().__init__()

    def create_model(self):
        """Creates the Scikit-learn model.

        Returns:
            tuple containing:
                - model: The Scikit-learn model.
                - model_name (str): The name of the model.

        """
        model = RandomForestClassifier()
        model_name = 'rf_model'
        return model, model_name

class BinaryKNN(AbstractSklearn):
    """An implementation of a Scikit-learn K-nearest Neighbors model."""

    def __init__(self):
        """Initializes a Scikit-learn model."""
        super().__init__()

    def create_model(self):
        """Creates the Scikit-learn model.

        Returns:
            tuple containing:
                - model: The Scikit-learn model.
                - model_name (str): The name of the model.

        """
        model = KNeighborsClassifier(n_neighbors=1)
        model_name = 'knn_model'
        return model, model_name

class RatingKNN(AbstractSklearn):
    """An implementation of a Scikit-learn K-nearest Neighbors model."""

    def __init__(self):
        """Initializes a Scikit-learn model."""
        super().__init__()

    def create_model(self):
        """Creates the Scikit-learn model.

        Returns:
            tuple containing:
                - model: The Scikit-learn model.
                - model_name (str): The name of the model.

        """
        model = KNeighborsClassifier(n_neighbors=30)
        model_name = 'rating_knn'
        return model, model_name
