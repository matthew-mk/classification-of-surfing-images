from classes.cnn import *
from classes.sklearn import *
from classes.dataset_handler import *

"""

"""

CATEGORIES = ['unsurfable', 'surfable']
TEST_SIZE = 0.2
TRAIN_SEED = 3
VAL_SEED = 1
N_SPLITS = 3
CONFIG = {
    'image_height': 40,
    'image_width': 40,
    'color_mode': 'rgb',
    'batch_size': 16,
    'epochs': 50
}

# Create CNN Model
basic_cnn = CNN(CONFIG)
basic_cnn.create_cnn()

# Create SVM Model
basic_svm = Sklearn()
basic_svm.create_svm()

# Create RF Model
basic_rf = Sklearn()
basic_rf.create_rf()

# Create KNN Model
basic_knn = Sklearn()
basic_knn.create_knn()


def cnn_test():
    # Setup datasets for CNN models
    cnn_dataset_handler = CNNDatasetHandler(CONFIG)
    cnn_dataset_handler.create_dataset(['binary_1'], CATEGORIES)
    X_train, X_val, X_test, y_train, y_val, y_test = cnn_dataset_handler.train_test_split(TEST_SIZE, TRAIN_SEED,
                                                                                          VAL_SEED)
    # Train CNN model
    basic_cnn.train_model(X_train, X_val, y_train, y_val)
    basic_cnn.plot_training_loss()
    basic_cnn.plot_training_accuracy()

    # Test the CNN model that was trained
    basic_cnn.test_model(X_test, y_test)


def sklearn_test():
    # Setup datasets for Scikit-learn models
    sklearn_dataset_handler = SklearnDatasetHandler(CONFIG)
    sklearn_dataset_handler.create_dataset(['binary_1'], CATEGORIES)
    X_train, X_test, y_train, y_test = sklearn_dataset_handler.train_test_split(TEST_SIZE, TRAIN_SEED)

    # Train the Scikit-learn models, then test them
    models = [basic_svm, basic_rf, basic_knn]
    for model in models:
        model.train_model(X_train, y_train)
        model.test_model(X_test, y_test)


def cnn_kfold_test():
    # Set up datasets
    cnn_dataset_handler = CNNDatasetHandler(CONFIG)
    cnn_dataset_handler.create_dataset(['binary_1'], CATEGORIES)
    X, y = cnn_dataset_handler.get_X_and_y()

    # K-Fold Cross Validation
    basic_cnn.kfold_cross_validation(X, y, N_SPLITS, TEST_SIZE)


def sklearn_kfold_test():
    # Set up datasets
    sklearn_dataset_handler = SklearnDatasetHandler(CONFIG)
    sklearn_dataset_handler.create_dataset(['binary_1'], CATEGORIES)
    X, y = sklearn_dataset_handler.get_X_and_y()

    # K-Fold Cross Validation
    basic_svm.kfold_cross_validation(X, y, N_SPLITS)
    basic_rf.kfold_cross_validation(X, y, N_SPLITS)
    basic_knn.kfold_cross_validation(X, y, N_SPLITS)


def main():
    # sklearn_test()
    # cnn_test()
    # sklearn_kfold_test()
    cnn_kfold_test()


if __name__ == '__main__':
    main()
