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
K_FOLD_SPLITS = 3
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
        'epochs': 100
    },
    'sklearn': {
        'image_height': 128,
        'image_width': 128,
        'color_mode': 'rgb'
    }
}


def train_and_test_model(model, seed, datasets_to_load, categories, test_size, configs, save_name=None):
    """Trains and tests a model using images from the specified datasets.

    Args:
        model: The model to be trained and tested.
        seed (int): The seed that the model will be trained on. A seed is a reproducible shuffle of the dataset.
        datasets_to_load (list[str]): A list containing the names of the datasets to be loaded. The model will be
            trained and tested using images from all of the specified datasets.
        categories (list[str]): The categories that the images in the datasets are categorized into.
        test_size (float): The proportion of the dataset that will be used for testing. Ranges from 0-1.
            E.g. 0.2 means 20% of images will be used for testing.
        configs (dict): Configuration settings used to set up the datasets for the Scikit-learn models and CNN models.
            Also includes training settings for CNN models.
        save_name (str): Optional parameter, defaults to None. If included, the trained model will be saved using this
            name.

    """
    if issubclass(type(model), BaseCNN):
        # The model is a CNN model
        # Setup datasets
        cnn_dataset_handler = CNNDatasetHandler(configs['cnn'])
        cnn_dataset_handler.create_dataset(datasets_to_load, categories, print_info=True)
        X_train, X_val, X_test, y_train, y_val, y_test = cnn_dataset_handler.train_test_split(test_size,
                                                                                              seed,
                                                                                              1)
        # Train CNN model
        model.train_model(X_train, X_val, y_train, y_val, save_name=save_name)
        model.plot_training_loss()
        model.plot_training_accuracy()
        # Test the CNN model that was trained
        model.test_model(X_test, y_test)
    elif issubclass(type(model), BaseSklearn):
        # The model is a Scikit-learn model
        # Setup datasets
        sklearn_dataset_handler = SklearnDatasetHandler(configs['sklearn'])
        sklearn_dataset_handler.create_dataset(datasets_to_load, categories, print_info=True)
        # Train and test the model
        X_train, X_test, y_train, y_test = sklearn_dataset_handler.train_test_split(test_size, seed)
        model.train_model(X_train, y_train)
        model.test_model(X_test, y_test)
        if save_name is not None:
            model.save_model(save_name)
    else:
        # The model is not a Scikit-learn or CNN model
        raise ValueError("An invalid model was given as input. It must be a Scikit-learn or CNN model.")


def train_and_test_models(models_and_seeds, datasets_to_load, categories, test_size, configs):
    """Trains and tests multiple models using images from the specified datasets.

    Args:
        models_and_seeds ((model, int)): The models and the seeds they will be trained/tested on.
        datasets_to_load (list[str]): A list containing the names of the datasets to be loaded. The model will be
            trained and tested using images from all of the specified datasets.
        categories (list[str]): The categories that the images in the datasets are categorized into.
        test_size (float): The proportion of the dataset that will be used for testing. Ranges from 0-1.
            E.g. 0.2 means 20% of images will be used for testing.
        configs (dict): Configuration settings used to set up the datasets for the Scikit-learn models and CNN models.
            Also includes training settings for CNN models.

    """
    for (model, seed) in models_and_seeds:
        train_and_test_model(model, seed, datasets_to_load, categories, test_size, configs)


def k_fold_cross_validation(models, datasets_to_load, categories, num_splits, configs):
    """Performs k-fold cross validation on models. 80% of the images are used for training and 20% for testing.

    Args:
        models (list): The models to perform k-fold cross validation on.
        datasets_to_load (list[str]): A list containing the names of the datasets to be loaded.
        categories (list[str]): The categories that the images in the datasets are categorized into.
        num_splits (int): The number of different ways in which the dataset will be split for k-fold cross
            validation.
        configs (dict): Configuration settings used to set up the datasets for the Scikit-learn models and CNN models.
            Also includes training settings for CNN models.

    """
    for model in models:
        if issubclass(type(model), BaseCNN):
            # The model is a CNN model
            # Set up datasets
            cnn_dataset_handler = CNNDatasetHandler(configs['cnn'])
            cnn_dataset_handler.create_dataset(datasets_to_load, categories, print_info=True)
            X, y = cnn_dataset_handler.get_X_and_y()
            # K-Fold Cross Validation
            model.kfold_cross_validation(X, y, num_splits)
        elif issubclass(type(model), BaseSklearn):
            # The model is a Scikit-learn model
            # Set up datasets
            sklearn_dataset_handler = SklearnDatasetHandler(configs['sklearn'])
            sklearn_dataset_handler.create_dataset(datasets_to_load, categories, print_info=True)
            X, y = sklearn_dataset_handler.get_X_and_y()
            # K-Fold Cross Validation
            model.kfold_cross_validation(X, y, num_splits)
        else:
            # The model is not a Scikit-learn or CNN model
            raise ValueError("An invalid model was given as input. Each model must be a Scikit-learn or CNN model.")


def find_best_sklearn_seeds(models, datasets_to_load, categories, test_size, configs, max_seed):
    """Tests Scikit-learn models on a variable number of seeds and prints the best seeds that were found.

    Args:
        models (list): The Scikit-learn models to find the best seeds for.
        datasets_to_load (list[str]): A list containing the names of the datasets to be loaded.
        categories (list[str]): The categories that the images in the datasets are categorized into.
        test_size (float): The proportion of the dataset that will be used for testing. Ranges from 0-1.
            E.g. 0.2 means 20% of images will be used for testing.
        configs (dict): Configuration settings used to set up the datasets for the Scikit-learn models and CNN models.
            Also includes training settings for CNN models.
        max_seed (int): The number of seeds to be tested. E.g. 100 tests up to seed 100.

    """
    # Setup datasets
    sklearn_dataset_handler = SklearnDatasetHandler(configs['sklearn'])
    sklearn_dataset_handler.create_dataset(datasets_to_load, categories, print_info=True)

    # Setup a list to contain the best seed and accuracy for each model
    models_and_info = []
    for model in models:
        models_and_info.append((model, 0, 0))

    # Find the best seeds
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

    # Print the best seeds that were found for each model
    print('-----------------------------------------------\n\nFinal results:')
    for tuple_index, (model, best_seed, best_acc) in enumerate(models_and_info):
        model.print_model_name()
        print(f"seed={best_seed}, accuracy={best_acc}%")


def test_saved_basic_models(configs):
    """Loads the models that were trained on images from a single surfing location and tests their performance on images
    from other locations.

    Args:
        configs (dict): Configuration settings used to set up the datasets for the Scikit-learn models and CNN models.
            Also includes training settings for CNN models.

    """
    # Load the models
    loaded_cnn = LoadedCNN(configs['cnn'])
    loaded_cnn.load_model('basic_cnn')

    loaded_svm = LoadedSklearn()
    loaded_svm.load_model('basic_svm')

    loaded_rf = LoadedSklearn()
    loaded_rf.load_model('basic_rf')

    loaded_knn = LoadedSklearn()
    loaded_knn.load_model('basic_knn')

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
    # Create the models
    cnn = CNN(CONFIGS['cnn'])
    svm = SVM()
    rf = RF()
    knn = KNN()

    models = [cnn]
    models_and_seeds = [(cnn, BEST_SEEDS['cnn']), (svm, BEST_SEEDS['svm']), (rf, BEST_SEEDS['rf']),
                        (knn, BEST_SEEDS['knn'])]

    # train_and_test_model(cnn, BEST_SEEDS['cnn'], DATASETS_TO_LOAD, CATEGORIES, TEST_SIZE, CONFIGS)
    # train_and_test_models(models_and_seeds, DATASETS_TO_LOAD, CATEGORIES, TEST_SIZE, CONFIGS)
    # k_fold_cross_validation(models, DATASETS_TO_LOAD, CATEGORIES, K_FOLD_SPLITS, CONFIGS)
    # find_best_sklearn_seeds(models, DATASETS_TO_LOAD, CATEGORIES, TEST_SIZE, CONFIGS, NUM_SEEDS_TO_TEST)
    # test_saved_basic_models(CONFIGS)

if __name__ == '__main__':
    main()
