"""This module trains models to classify images of surfing locations based on whether the conditions in the images are
suitable for surfing or not (binary classification). The models can be trained using images from 1 to 5 surfing
locations."""

from classes.cnn import *
from classes.sklearn import *
from classes.dataset_handler import *

DATASET_TYPE = 'binary'
CATEGORIES = ['unsurfable', 'surfable']
NUM_LOCATIONS = 5
TEST_SIZE = 0.2
KFOLD_SPLITS = 5
NUM_SEEDS_TO_TEST = 10
DATASETS_TO_LOAD = get_dataset_names(DATASET_TYPE, NUM_LOCATIONS)
BEST_SEEDS = {
    'cnn': get_seed([3, 0, 0, 0, 123], NUM_LOCATIONS),
    'svm': get_seed([3, 88, 3, 47, 8], NUM_LOCATIONS),
    'rf': get_seed([3, 72, 30, 13, 73], NUM_LOCATIONS),
    'knn': get_seed([3, 50, 1, 26, 73], NUM_LOCATIONS)
}
CONFIGS = {
    'cnn': {
        'image_height': 40,
        'image_width': 40,
        'color_mode': 'rgb',
        'batch_size': 16,
        'epochs': 150
    },
    'sklearn': {
        'image_height': 128,
        'image_width': 128,
        'color_mode': 'rgb'
    }
}


def train_and_test_cnn(datasets_to_load, categories, test_size, best_seeds, configs):
    # Setup datasets
    cnn_dataset_handler = CNNDatasetHandler(configs['cnn'])
    cnn_dataset_handler.create_dataset(datasets_to_load, categories, print_info=True)
    X_train, X_val, X_test, y_train, y_val, y_test = cnn_dataset_handler.train_test_split(test_size,
                                                                                          best_seeds['cnn'],
                                                                                          1)

    # Create the CNN model
    cnn = CNN(configs['cnn'])

    # Train CNN model
    cnn.train_model(X_train, X_val, y_train, y_val)
    cnn.plot_training_loss()
    cnn.plot_training_accuracy()

    # Test the CNN model that was trained
    cnn.test_model(X_test, y_test)

def train_and_test_sklearn(datasets_to_load, categories, test_size, best_seeds, configs):
    # Setup datasets
    sklearn_dataset_handler = SklearnDatasetHandler(configs['sklearn'])
    sklearn_dataset_handler.create_dataset(datasets_to_load, categories, print_info=True)

    # Create the Scikit-learn models
    svm = SVM()
    rf = RF()
    knn = KNN()

    # Get the best training seeds that have been found for these algorithms based on the number of locations
    # that are used
    svm_seed = best_seeds['svm']
    rf_seed = best_seeds['rf']
    knn_seed = best_seeds['knn']

    # Train and test the Scikit-learn models
    X_train, X_test, y_train, y_test = sklearn_dataset_handler.train_test_split(test_size, svm_seed)
    svm.train_model(X_train, y_train)
    svm.test_model(X_test, y_test)
    # svm.save_model('basic_svm_2')  # uncomment to save the SVM model
    X_train, X_test, y_train, y_test = sklearn_dataset_handler.train_test_split(test_size, rf_seed)
    rf.train_model(X_train, y_train)
    rf.test_model(X_test, y_test)
    # rf.save_model('basic_rf_2')  # uncomment to save the SVM model
    X_train, X_test, y_train, y_test = sklearn_dataset_handler.train_test_split(test_size, knn_seed)
    knn.train_model(X_train, y_train)
    knn.test_model(X_test, y_test)
    # knn.save_model('basic_knn_2')  # uncomment to save the KNN model

def find_best_sklearn_seeds(datasets_to_load, categories, test_size, configs, max_seed):
    # Setup datasets
    sklearn_dataset_handler = SklearnDatasetHandler(configs['sklearn'])
    sklearn_dataset_handler.create_dataset(datasets_to_load, categories, print_info=True)

    # Create the Scikit-learn models
    svm = SVM()
    rf = RF()
    knn = KNN()

    # Find the best seeds
    models_and_info = [(svm, 0, 0), (rf, 0, 0), (knn, 0, 0)]
    for seed in range(0, max_seed + 1):
        print('\n-----------------------------------------------')
        print(f'\nSeed {seed} results:')

        # Shuffle the dataset using the current seed
        X_train, X_test, y_train, y_test = sklearn_dataset_handler.train_test_split(test_size, seed)

        # Train and test the models
        for tuple_index, (model, best_seed, best_acc) in enumerate(models_and_info):
            model.train_model(X_train, y_train)
            acc = model.test_model(X_test, y_test, return_acc=True)
            if acc > best_acc:
                temp = list(models_and_info[tuple_index])
                temp[1], temp[2] = seed, acc
                models_and_info[tuple_index] = tuple(temp)

    print(f"\nSVM: seed={models_and_info[0][1]}, accuracy={models_and_info[0][2]}%")
    print(f"RF: seed={models_and_info[1][1]}, accuracy={models_and_info[1][2]}%")
    print(f"KNN: seed={models_and_info[2][1]}, accuracy={models_and_info[2][2]}%")

def cnn_kfold(datasets_to_load, categories, test_size, num_splits, configs):
    # Set up datasets
    cnn_dataset_handler = CNNDatasetHandler(configs['cnn'])
    cnn_dataset_handler.create_dataset(datasets_to_load, categories, print_info=True)
    X, y = cnn_dataset_handler.get_X_and_y()

    # Create the CNN model
    cnn = CNN(configs['cnn'])

    # K-Fold Cross Validation
    cnn.kfold_cross_validation(X, y, num_splits, test_size)

def sklearn_kfold(datasets_to_load, categories, num_splits, configs):
    # Set up datasets
    sklearn_dataset_handler = SklearnDatasetHandler(configs['sklearn'])
    sklearn_dataset_handler.create_dataset(datasets_to_load, categories, print_info=True)
    X, y = sklearn_dataset_handler.get_X_and_y()

    # Create the Scikit-learn models
    svm = SVM()
    rf = RF()
    knn = KNN()

    # K-Fold Cross Validation
    svm.kfold_cross_validation(X, y, num_splits)
    rf.kfold_cross_validation(X, y, num_splits)
    knn.kfold_cross_validation(X, y, num_splits)

def test_saved_basic_models(configs):
    # Load the 'basic' models
    loaded_cnn = LoadedCNN(configs['cnn'], 'basic_cnn')
    loaded_svm = LoadedSklearn('basic_svm')
    loaded_rf = LoadedSklearn('basic_rf')
    loaded_knn = LoadedSklearn('basic_knn')

    # Setup datasets for the loaded CNN model
    cnn_dataset_handler = CNNDatasetHandler(configs['cnn'])
    cnn_dataset_handler.create_dataset(['binary_1'], ['unsurfable', 'surfable'])
    X_train, X_val, X_test, y_train, y_val, y_test = cnn_dataset_handler.train_test_split(0.2, 3, 1)

    # Test the loaded CNN model
    print('\nPerformance on binary_1 dataset (the one the models were trained on):')
    loaded_cnn.test_model(X_test, y_test)

    # Setup datasets for the loaded Scikit-learn models
    sklearn_dataset_handler = SklearnDatasetHandler(configs['sklearn'])
    sklearn_dataset_handler.create_dataset(['binary_1'], ['unsurfable', 'surfable'])
    X_train, X_test, y_train, y_test = sklearn_dataset_handler.train_test_split(0.2, 3)

    # Test the loaded Scikit-learn models on the dataset that they were trained on
    loaded_svm.test_model(X_test, y_test)
    loaded_rf.test_model(X_test, y_test)
    loaded_knn.test_model(X_test, y_test)

    # Test the models on the datasets that they have not been trained on
    for i in range(2, 6):
        print("\n-----------------------------------------")
        print(f'\nPerformance on binary_{i} dataset:')

        # Set up dataset for CNN model
        cnn_dataset_handler.create_dataset([f'binary_{i}'], ['unsurfable', 'surfable'])

        # Get the images and labels
        X_test, y_test = cnn_dataset_handler.get_X_and_y()

        # Test CNN model
        loaded_cnn.test_model(X_test, y_test)

        # Set up dataset for Scikit-learn models
        sklearn_dataset_handler.create_dataset([f'binary_{i}'], ['unsurfable', 'surfable'])
        X_test, y_test = sklearn_dataset_handler.get_X_and_y()

        # Test Scikit-learn models
        loaded_svm.test_model(X_test, y_test)
        loaded_rf.test_model(X_test, y_test)
        loaded_knn.test_model(X_test, y_test)

def main():
    # train_and_test_cnn(DATASETS_TO_LOAD, CATEGORIES, TEST_SIZE, BEST_SEEDS, CONFIGS)
    # train_and_test_sklearn(DATASETS_TO_LOAD, CATEGORIES, TEST_SIZE, BEST_SEEDS, CONFIGS)
    # find_best_sklearn_seeds(DATASETS_TO_LOAD, CATEGORIES, TEST_SIZE, CONFIGS, NUM_SEEDS_TO_TEST)
    # cnn_kfold(DATASETS_TO_LOAD, CATEGORIES, TEST_SIZE, KFOLD_SPLITS, CONFIGS)
    sklearn_kfold(DATASETS_TO_LOAD, CATEGORIES, KFOLD_SPLITS, CONFIGS)
    # test_saved_basic_models(CONFIGS)

if __name__ == '__main__':
    main()
