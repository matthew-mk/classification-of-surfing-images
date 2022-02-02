"""This module trains models to classify images based on whether they contain suitable conditions for surfing or not
(binary classification). The models are trained using images from a single surfing location and evaluated using
appropriate metrics."""

from classes.cnn import *
from classes.sklearn import *
from classes.dataset_handler import *

CATEGORIES = ['unsurfable', 'surfable']
TEST_SIZE = 0.2
TRAIN_SEED = 3
VAL_SEED = 1
KFOLD_SPLITS = 5
CNN_CONFIG = {
    'image_height': 40,
    'image_width': 40,
    'color_mode': 'rgb',
    'batch_size': 16,
    'epochs': 300
}
SKLEARN_CONFIG = {
    'image_height': 128,
    'image_width': 128,
    'color_mode': 'rgb'
}

def get_basic_models(modelType=None):
    if modelType == 'cnn':
        basic_cnn = CNN(CNN_CONFIG)
        basic_cnn.create_basic_cnn()
        return basic_cnn
    elif modelType == 'sklearn':
        basic_svm = Sklearn()
        basic_svm.create_basic_svm()
        basic_rf = Sklearn()
        basic_rf.create_basic_rf()
        basic_knn = Sklearn()
        basic_knn.create_basic_knn()
        return basic_svm, basic_rf, basic_knn
    else:
        basic_cnn = CNN(CNN_CONFIG)
        basic_cnn.create_basic_cnn()
        basic_svm = Sklearn()
        basic_svm.create_basic_svm()
        basic_rf = Sklearn()
        basic_rf.create_basic_rf()
        basic_knn = Sklearn()
        basic_knn.create_basic_knn()
        return basic_cnn, basic_svm, basic_rf, basic_knn


def train_and_test_cnn():
    # Setup datasets for the CNN model
    cnn_dataset_handler = CNNDatasetHandler(CNN_CONFIG)
    cnn_dataset_handler.create_dataset(['binary_1'], CATEGORIES)
    X_train, X_val, X_test, y_train, y_val, y_test = cnn_dataset_handler.train_test_split(TEST_SIZE, TRAIN_SEED,
                                                                                          VAL_SEED)

    # Get the CNN model
    basic_cnn = get_basic_models(modelType='cnn')

    # Train CNN model
    basic_cnn.train_model(X_train, X_val, y_train, y_val)
    basic_cnn.plot_training_loss()
    basic_cnn.plot_training_accuracy()

    # Test the CNN model that was trained
    basic_cnn.test_model(X_test, y_test)


def train_and_test_sklearn():
    # Setup datasets for Scikit-learn models
    sklearn_dataset_handler = SklearnDatasetHandler(SKLEARN_CONFIG)
    sklearn_dataset_handler.create_dataset(['binary_1'], CATEGORIES)
    X_train, X_test, y_train, y_test = sklearn_dataset_handler.train_test_split(TEST_SIZE, TRAIN_SEED)

    # Get the Scikit-learn models
    basic_svm, basic_rf, basic_knn = get_basic_models(modelType='sklearn')
    models = [basic_svm, basic_rf, basic_knn]

    # Train the Scikit-learn models, then test them
    for model in models:
        model.train_model(X_train, y_train)
        model.test_model(X_test, y_test)
        model.classification_report(X_test, y_test)


def cnn_kfold():
    # Set up datasets
    cnn_dataset_handler = CNNDatasetHandler(CNN_CONFIG)
    cnn_dataset_handler.create_dataset(['binary_1'], CATEGORIES)
    X, y = cnn_dataset_handler.get_X_and_y()

    # Get the CNN model
    basic_cnn = get_basic_models(modelType='cnn')

    # K-Fold Cross Validation
    basic_cnn.kfold_cross_validation(X, y, KFOLD_SPLITS, TEST_SIZE)


def sklearn_kfold():
    # Set up datasets
    sklearn_dataset_handler = SklearnDatasetHandler(SKLEARN_CONFIG)
    sklearn_dataset_handler.create_dataset(['binary_1'], CATEGORIES)
    X, y = sklearn_dataset_handler.get_X_and_y()

    # Get the Scikit-learn models
    basic_svm, basic_rf, basic_knn = get_basic_models(modelType='sklearn')

    # K-Fold Cross Validation
    basic_svm.kfold_cross_validation(X, y, KFOLD_SPLITS)
    basic_rf.kfold_cross_validation(X, y, KFOLD_SPLITS)
    basic_knn.kfold_cross_validation(X, y, KFOLD_SPLITS)


def test_saved_models():
    # Load 'basic_cnn' model
    loaded_cnn = CNN(CNN_CONFIG)
    loaded_cnn.load_model('basic_cnn')

    # Load 'basic_svm' model
    loaded_svm = Sklearn()
    loaded_svm.load_model('basic_svm')

    # Load 'basic_rf' model
    loaded_rf = Sklearn()
    loaded_rf.load_model('basic_rf')

    # Load 'basic_knn' model
    loaded_knn = Sklearn()
    loaded_knn.load_model('basic_knn')

    # Setup datasets for the loaded CNN model
    cnn_dataset_handler = CNNDatasetHandler(CNN_CONFIG)
    cnn_dataset_handler.create_dataset(['binary_1'], CATEGORIES)
    X_train, X_val, X_test, y_train, y_val, y_test = cnn_dataset_handler.train_test_split(TEST_SIZE, TRAIN_SEED,
                                                                                          VAL_SEED)
    # Test the loaded CNN model
    loaded_cnn.test_model(X_test, y_test)

    # Setup datasets for the loaded Scikit-learn models
    sklearn_dataset_handler = SklearnDatasetHandler(SKLEARN_CONFIG)
    sklearn_dataset_handler.create_dataset(['binary_1'], CATEGORIES)
    X_train, X_test, y_train, y_test = sklearn_dataset_handler.train_test_split(TEST_SIZE, TRAIN_SEED)

    # Test the loaded Scikit-learn models
    loaded_svm.test_model(X_test, y_test)
    loaded_rf.test_model(X_test, y_test)
    loaded_knn.test_model(X_test, y_test)


def main():
    # train_and_test_cnn()
    # train_and_test_sklearn()
    # cnn_kfold()
    # sklearn_kfold()
    test_saved_models()


if __name__ == '__main__':
    main()
