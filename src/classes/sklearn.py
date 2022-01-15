from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import pickle


class SKLearn:
    """

    """
    def __init__(self, model):
        """The constructor for a SciKit Learn model.

        Args:
            model (): The model to be used by this class.

        """
        self.model_name = 'sklearn_model'
        self.model = model

    def get_model(self):
        """Returns the SciKit Learn model.

        Returns:
            The SciKit Learn model.

        """
        return self.model

    def get_model_name(self):
        """Returns the name of the current model.

        Returns:
            The name of the model.

        """
        return self.model_name

    def train_model(self, X_train, y_train):
        """Trains the model using a training dataset.

        Args:
            X_train (list[list]): The training dataset images, where each image is represented as a pixel array.
            y_train (list[int]): The labels of the images in the dataset.

        """
        self.model.fit(X_train, y_train)

    def test_model(self, X_test, y_test):
        """Tests the model on a dataset and prints the accuracy.

        Args:
            X_test (list[list]): The images in the dataset, where each image is represented as a pixel array.
            y_test (list[int]): The labels of the images in the dataset.

        """
        acc = self.model.score(X_test, y_test)
        print('Accuracy: {}%'.format(round(acc * 100, 2)))
        print('Number of correct predictions: {}/{}'.format(round(len(X_test) * acc), len(X_test)))

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

    def kfold_cross_validation(self, X, y, folds):
        """K-Fold Cross Validation is applied to the model and info about accuracy, precision, and recall is printed.

        Note: This function will only return legitimate results if the model has not been trained on the dataset that is
        used.

        Args:
            X (list[list]): The images in the dataset, where each image is represented as a list of pixel values.
            y (list[int]): The labels of the images.
            folds (int): The number of folds the dataset will be divided into.

        """
        scoring_types = ['accuracy', 'precision', 'recall', 'f1']
        cv_results = cross_validate(self.model, X, y, cv=folds, scoring=scoring_types)
        for scoring_type in scoring_types:
            cv_result = cv_results['test_{}'.format(scoring_type)]
            print('Max {}: {}% '.format(scoring_type, round(np.max(cv_result) * 100, 2)))
            print('Min {}: {}% '.format(scoring_type, round(np.min(cv_result) * 100, 2)))
            print('Average {}: {}% '.format(scoring_type, round(np.average(cv_result) * 100, 2)))