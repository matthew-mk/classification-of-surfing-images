from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import pickle


class Sklearn:
    """A class for creating, training, and evaluating a Scikit-learn model."""
    def __init__(self):
        """Initializes a Scikit-learn model."""
        self.model = None
        self.model_name = 'unnamed_model'

    def get_model(self):
        """Returns the Scikit-learn model.

        Returns:
            The Scikit-learn model.

        """
        return self.model

    def print_model_name(self):
        """Prints the name of the current model."""
        print("\n" + self.model_name)

    def create_svm_1(self):
        """Creates an implementation of a Scikit-learn SVM model where C=4."""
        self.model = SVC(C=4)
        self.model_name = 'basic_svm'

    def create_rf_1(self):
        """Creates an implementation of a Scikit-learn Random Forest model."""
        self.model = RandomForestClassifier()
        self.model_name = 'basic_rf'

    def create_knn_1(self):
        """Creates an implementation of a Scikit-learn KNN model which uses 1 neighbour."""
        self.model = KNeighborsClassifier(n_neighbors=1)
        self.model_name = 'basic_knn'

    def create_svm_2(self):
        """Creates an implementation of a Scikit-learn SVM model where C=1."""
        self.model = SVC(C=4)
        self.model_name = 'svm_2'

    def create_rf_2(self):
        """Creates an implementation of a Scikit-learn Random Forest model."""
        self.model = RandomForestClassifier()
        self.model_name = 'rf_2'

    def create_knn_2(self):
        """Creates an implementation of a Scikit-learn KNN model which uses 1 neighbour."""
        self.model = KNeighborsClassifier(n_neighbors=1)
        self.model_name = 'knn_2'

    def create_custom_model(self, model, model_name):
        """Allows a custom Scikit-learn model to be passed in as a parameter.

        Args:
            model: The Scikit-learn model.
            model_name (str): The name of the model.

        """
        self.model = model
        self.model_name = model_name

    def train_model(self, X_train, y_train):
        """Trains the model using a training dataset.

        Args:
            X_train (list[list]): The training dataset images, where each image is represented as a pixel array.
            y_train (list[int]): The labels of the images in the dataset.

        """
        self.model.fit(X_train, y_train)

    def test_model(self, X_test, y_test, show_report=False):
        """Tests the model on a dataset and prints the accuracy.

        Args:
            X_test (list[list]): The images in the dataset, where each image is represented as a pixel array.
            y_test (list[int]): The labels of the images in the dataset.
            show_report (bool): Optional variable, defaults to False. If set to true it will print a classification
              report that has details about accuracy, precision, and recall.

        """
        acc = self.model.score(X_test, y_test)
        self.print_model_name()
        print('Accuracy: {}%'.format(round(acc * 100, 2)))
        print('Number of correct predictions: {}/{}'.format(round(len(X_test) * acc), len(X_test)))
        if show_report:
            y_predicted = self.model.predict(X_test)
            print("Classification report: ")
            print(classification_report(y_test, y_predicted))

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
        pickle.dump(self.model, open('../saved_models/{}.sav'.format(model_name), 'wb'))
        self.model_name = model_name

    def load_model(self, model_name):
        """Loads a model from the 'saved_models' folder and sets it to be the model.

        Args:
            model_name (str): The name of the model to be loaded, e.g. 'basic_svm'.

        """
        self.model = pickle.load(open('../saved_models/{}.sav'.format(model_name), 'rb'))
        self.model_name = model_name

    def kfold_cross_validation(self, X, y, n_splits):
        """K-Fold Cross Validation is applied to the model and info about accuracy, precision, and recall is printed.

        Note: This function will only return legitimate results if the model has not been trained on the dataset that is
        used.

        Args:
            X (list[list]): The images in the dataset, where each image is represented as a list of pixel values.
            y (list[int]): The labels of the images.
            n_splits (int): The number of folds the dataset will be divided into.

        """
        print(f'{n_splits}-Fold Cross Validation: {self.model_name}')
        scoring_types = ['accuracy', 'precision', 'recall', 'f1']
        cv_results = cross_validate(self.model, X, y, cv=n_splits, scoring=scoring_types)
        for scoring_type in scoring_types:
            cv_result = cv_results['test_{}'.format(scoring_type)]
            print('Max {}: {}% '.format(scoring_type, round(np.max(cv_result) * 100, 2)))
            print('Min {}: {}% '.format(scoring_type, round(np.min(cv_result) * 100, 2)))
            print('Average {}: {}% '.format(scoring_type, round(np.average(cv_result) * 100, 2)))
