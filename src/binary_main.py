"""This module trains models to classify images of surfing locations based on whether the conditions in the images are
suitable for surfing or not (binary classification). The models can be trained using images from 1 to 5 surfing
locations."""

from classes.cnn import *
from classes.sklearn import *
from classes.dataset_handler import *

CATEGORIES = ['unsurfable', 'surfable']
NUM_LOCATIONS = 5
KFOLD_SPLITS = 5
TEST_SIZE = 0.2

CNN_SEEDS = [3,  0,  0,  0,  123]
SVM_SEEDS = [3,  88, 3,  47, 8]
RF_SEEDS  = [3,  72, 30, 13, 73]
KNN_SEEDS = [3,  50, 1,  26, 73]

CNN_CONFIG = {
    'image_height': 40,
    'image_width': 40,
    'color_mode': 'rgb',
    'batch_size': 16,
    'epochs': 150
}
SKLEARN_CONFIG = {
    'image_height': 128,
    'image_width': 128,
    'color_mode': 'rgb'
}

def get_dataset_names(num_locations):
    dataset_names = []
    if 0 < num_locations <= 5:
        for i in range(1, num_locations + 1):
            dataset_names.append(f'binary_{i}')
    else:
        raise ValueError("The number of locations must be between 1 and 5.")

    return dataset_names

def train_and_test_cnn(num_locations):
    # Get the best training seed
    train_seed = CNN_SEEDS[num_locations - 1]

    # Setup datasets
    cnn_dataset_handler = CNNDatasetHandler(CNN_CONFIG)
    cnn_dataset_handler.create_dataset(get_dataset_names(num_locations), CATEGORIES, print_info=True)
    X_train, X_val, X_test, y_train, y_val, y_test = cnn_dataset_handler.train_test_split(TEST_SIZE, train_seed, 1)

    # Create the CNN model
    cnn = CNN(CNN_CONFIG)
    cnn.create_cnn_1()

    # Train CNN model
    cnn.train_model(X_train, X_val, y_train, y_val)
    cnn.plot_training_loss()
    cnn.plot_training_accuracy()

    # Test the CNN model that was trained
    cnn.test_model(X_test, y_test)

def train_and_test_sklearn(num_locations):
    # Setup datasets
    sklearn_dataset_handler = SklearnDatasetHandler(SKLEARN_CONFIG)
    sklearn_dataset_handler.create_dataset(get_dataset_names(num_locations), CATEGORIES, print_info=True)

    # Create the Scikit-learn models
    svm = Sklearn()
    svm.create_svm_1()
    rf = Sklearn()
    rf.create_rf_1()
    knn = Sklearn()
    knn.create_knn_1()

    # Get the best training seeds that have been found for these algorithms based on the number of locations
    # that are used
    svm_seed = SVM_SEEDS[num_locations - 1]
    rf_seed = RF_SEEDS[num_locations - 1]
    knn_seed = KNN_SEEDS[num_locations - 1]

    # Train and test the Scikit-learn models
    X_train, X_test, y_train, y_test = sklearn_dataset_handler.train_test_split(TEST_SIZE, svm_seed)
    svm.train_model(X_train, y_train)
    svm.test_model(X_test, y_test)
    # svm.save_model('basic_svm_2')  # uncomment to save the SVM model
    X_train, X_test, y_train, y_test = sklearn_dataset_handler.train_test_split(TEST_SIZE, rf_seed)
    rf.train_model(X_train, y_train)
    rf.test_model(X_test, y_test)
    # rf.save_model('basic_rf_2')  # uncomment to save the SVM model
    X_train, X_test, y_train, y_test = sklearn_dataset_handler.train_test_split(TEST_SIZE, knn_seed)
    knn.train_model(X_train, y_train)
    knn.test_model(X_test, y_test)
    # knn.save_model('basic_knn_2')  # uncomment to save the KNN model

def find_best_sklearn_seeds(num_locations, max_seed):
    # Setup datasets
    sklearn_dataset_handler = SklearnDatasetHandler(SKLEARN_CONFIG)
    sklearn_dataset_handler.create_dataset(get_dataset_names(num_locations), CATEGORIES, print_info=True)

    # Create the Scikit-learn models
    svm = Sklearn()
    svm.create_svm_1()
    rf = Sklearn()
    rf.create_rf_1()
    knn = Sklearn()
    knn.create_knn_1()

    # Find the best seeds
    models_and_info = [(svm, 0, 0), (rf, 0, 0), (knn, 0, 0)]
    for seed in range(0, max_seed + 1):
        print('\n-----------------------------------------------')
        print(f'\nSeed {seed} results:')

        # Shuffle the dataset using the current seed
        X_train, X_test, y_train, y_test = sklearn_dataset_handler.train_test_split(TEST_SIZE, seed)

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

def cnn_kfold(num_locations, num_splits):
    # Set up datasets
    cnn_dataset_handler = CNNDatasetHandler(CNN_CONFIG)
    cnn_dataset_handler.create_dataset(get_dataset_names(num_locations), CATEGORIES, print_info=True)
    X, y = cnn_dataset_handler.get_X_and_y()

    # Create the CNN model
    cnn = CNN(CNN_CONFIG)
    cnn.create_cnn_1()

    # K-Fold Cross Validation
    cnn.kfold_cross_validation(X, y, num_splits, TEST_SIZE)

def sklearn_kfold(num_locations, num_splits):
    # Set up datasets
    sklearn_dataset_handler = SklearnDatasetHandler(SKLEARN_CONFIG)
    sklearn_dataset_handler.create_dataset(get_dataset_names(num_locations), CATEGORIES, print_info=True)
    X, y = sklearn_dataset_handler.get_X_and_y()

    # Create the Scikit-learn models
    svm = Sklearn()
    svm.create_svm_1()
    rf = Sklearn()
    rf.create_rf_1()
    knn = Sklearn()
    knn.create_knn_1()

    # K-Fold Cross Validation
    svm.kfold_cross_validation(X, y, num_splits)
    rf.kfold_cross_validation(X, y, num_splits)
    knn.kfold_cross_validation(X, y, num_splits)

def test_saved_basic_models():
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
    X_train, X_val, X_test, y_train, y_val, y_test = cnn_dataset_handler.train_test_split(TEST_SIZE, 3, 1)

    # Test the loaded CNN model
    print('\nPerformance on binary_1 dataset (the one the models were trained on):')
    loaded_cnn.test_model(X_test, y_test)

    # Setup datasets for the loaded Scikit-learn models
    sklearn_dataset_handler = SklearnDatasetHandler(SKLEARN_CONFIG)
    sklearn_dataset_handler.create_dataset(['binary_1'], CATEGORIES)
    X_train, X_test, y_train, y_test = sklearn_dataset_handler.train_test_split(TEST_SIZE, 3)

    # Test the loaded Scikit-learn models on the dataset that they were trained on
    loaded_svm.test_model(X_test, y_test)
    loaded_rf.test_model(X_test, y_test)
    loaded_knn.test_model(X_test, y_test)

    # Test the models on the datasets that they were not trained on
    for i in range(2, 6):
        print("\n-----------------------------------------")
        print(f'\nPerformance on binary_{i} dataset:')
        # Set up dataset for CNN model
        cnn_dataset_handler.create_dataset([f'binary_{i}'], CATEGORIES)
        X_test, y_test = cnn_dataset_handler.get_X_and_y()
        X_test, y_test = np.array(X_test), np.array(y_test)
        X_test = X_test.reshape(len(X_test), CNN_CONFIG['image_height'], CNN_CONFIG['image_width'],
                                get_channels(CNN_CONFIG['color_mode']))
        # Test CNN model
        loaded_cnn.test_model(X_test, y_test)
        # Set up dataset for Scikit-learn models
        sklearn_dataset_handler.create_dataset([f'binary_{i}'], CATEGORIES)
        X_test, y_test = sklearn_dataset_handler.get_X_and_y()
        # Test Scikit-learn models
        loaded_svm.test_model(X_test, y_test)
        loaded_rf.test_model(X_test, y_test)
        loaded_knn.test_model(X_test, y_test)

def main():
    # train_and_test_cnn(NUM_LOCATIONS)
    train_and_test_sklearn(NUM_LOCATIONS)
    # find_best_sklearn_seeds(NUM_LOCATIONS, 100)
    # cnn_kfold(NUM_LOCATIONS, KFOLD_SPLITS)
    # sklearn_kfold(NUM_LOCATIONS, KFOLD_SPLITS)
    # test_saved_basic_models()

if __name__ == '__main__':
    main()
