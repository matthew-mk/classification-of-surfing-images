"""This module defines an abstract base class that contains common functionality for Scikit-learn models. There are also
subclasses that inherit from the base class, including particular implementations of SVM, RF, and KNN models. """

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_validate
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

    def plot_confusion_matrix(self, X_test, y_test):
        """Creates a confusion matrix showing the model's predictions versus what the correct answers were.

        Args:
            X_test (list[list]): The images in the dataset, where each image is represented as a pixel array.
            y_test (list[int]): The labels of the images in the dataset.

        """
        y_predicted = self.model.predict(X_test)
        cm = confusion_matrix(y_test, y_predicted)
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
            pickle.dump(self.model, open('../saved_models/{}.sav'.format(model_name), 'wb'))
            self.model_name = model_name
        else:
            print('The model could not be saved. An invalid name was used.')

    def kfold_cross_validation(self, X, y, n_splits):
        """K-Fold Cross Validation is applied to the model and info about accuracy, precision, and recall is printed.

        Note: This function will only return legitimate results if the model has not been trained on the dataset that is
        used.

        Args:
            X (list[list]): The images in the dataset, where each image is represented as a list of pixel values.
            y (list[int]): The labels of the images.
            n_splits (int): The number of folds the dataset will be divided into.

        """
        # Conduct K-Fold Cross Validation
        scoring_types = ['accuracy', 'precision', 'recall', 'f1']
        cv_results = cross_validate(self.model, X, y, cv=n_splits, scoring=scoring_types)

        # Change the formatting of all the results to have 2 decimal places
        for scoring_type in scoring_types:
            cv_result = cv_results[f'test_{scoring_type}']
            index = 0
            for _ in cv_result:
                cv_result[index] = np.round(cv_result[index] * 100, 2)
                index += 1

        accuracy_results = cv_results['test_accuracy']
        precision_results = cv_results['test_precision']
        recall_results = cv_results['test_recall']
        f1_results = cv_results['test_f1']

        # Print results from each fold
        print(f'\n{n_splits}-Fold Cross Validation: {self.model_name}')
        for i in range(0, n_splits):
            fold_accuracy = f'{accuracy_results[i]}%'
            fold_precision = f'{precision_results[i]}%'
            fold_recall = f'{recall_results[i]}%'
            fold_f1 = f1_results[i]
            print(f'Fold {i + 1}: accuracy={fold_accuracy}, precision={fold_precision}, ' +
                  f'recall={fold_recall}, f1={fold_f1}')

        # Overall results
        print("\nOverall stats: ")
        for scoring_type in scoring_types:
            if scoring_type == 'f1':
                ending = ''
            else:
                ending = '%'
            cv_result = cv_results[f'test_{scoring_type}']
            print(f'{scoring_type}: maximum={np.max(cv_result)}{ending}, minimum={np.min(cv_result)}{ending}, ' +
                  f'average={np.average(cv_result)}{ending}')


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
            self.model = pickle.load(open('../saved_models/{}.sav'.format(model_name), 'rb'))
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


class KNN(AbstractSklearn):
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
