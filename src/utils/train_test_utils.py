"""This module contains utility functions for training, testing, and evaluating models."""

from classes.cnn import *
from classes.sklearn import *
from classes.dataset_handler import *

def train_and_test_model(model, seed, enclosing_folder, datasets_to_load, test_size, configs, save_name=None):
    """Trains and tests a model using images from the specified datasets.

    Args:
        model: The model to be trained and tested.
        seed (int): The seed that the model will be trained on. A seed is a reproducible shuffle of the dataset.
        enclosing_folder (str): The name of the folder that contains the datasets to load. E.g. 'binary' or 'rating'.
        datasets_to_load (list[str]): A list containing the names of the datasets to be loaded. The model will be
            trained and tested using images from all of the specified datasets.
        test_size (float): The proportion of the dataset that will be used for testing. Ranges from 0-1.
            E.g. 0.2 means 20% of images will be used for testing.
        configs (dict): Configuration settings used to set up the datasets for the Scikit-learn models and CNN models.
            Also includes training settings for CNN models.
        save_name (str): Optional parameter, defaults to None. If included, the trained model will be saved using this
            name.

    """
    if issubclass(type(model), AbstractCNN):
        # The model is a CNN model
        # Setup datasets
        cnn_dataset_handler = CNNDatasetHandler(configs['cnn'])
        cnn_dataset_handler.create_dataset(enclosing_folder, datasets_to_load, print_info=True)
        X_train, X_val, X_test, y_train, y_val, y_test = cnn_dataset_handler.train_test_split(test_size,
                                                                                              seed,
                                                                                              1)
        # Train CNN model
        model.train_model(X_train, X_val, y_train, y_val, save_name=save_name)
        model.plot_training_loss()
        model.plot_training_accuracy()
        # Test the CNN model that was trained
        model.test_model(X_test, y_test)
    elif issubclass(type(model), AbstractSklearn):
        # The model is a Scikit-learn model
        # Setup datasets
        sklearn_dataset_handler = SklearnDatasetHandler(configs['sklearn'])
        sklearn_dataset_handler.create_dataset(enclosing_folder, datasets_to_load, print_info=True)
        # Train and test the model
        X_train, X_test, y_train, y_test = sklearn_dataset_handler.train_test_split(test_size, seed)
        model.train_model(X_train, y_train)
        model.test_model(X_test, y_test)
        if save_name is not None:
            model.save_model(save_name)
    else:
        # The model is not a Scikit-learn or CNN model
        raise ValueError("An invalid model was given as input. It must be a Scikit-learn or CNN model.")


def train_and_test_models(models_and_seeds, enclosing_folder, datasets_to_load, test_size, configs):
    """Trains and tests multiple models using images from the specified datasets.

    Args:
        models_and_seeds ((model, int)): The models and the seeds they will be trained/tested on.
        enclosing_folder (str): The name of the folder that contains the datasets to load. E.g. 'binary' or 'rating'.
        datasets_to_load (list[str]): A list containing the names of the datasets to be loaded. The model will be
            trained and tested using images from all of the specified datasets.
        test_size (float): The proportion of the dataset that will be used for testing. Ranges from 0-1.
            E.g. 0.2 means 20% of images will be used for testing.
        configs (dict): Configuration settings used to set up the datasets for the Scikit-learn models and CNN models.
            Also includes training settings for CNN models.

    """
    for (model, seed) in models_and_seeds:
        train_and_test_model(model, seed, enclosing_folder, datasets_to_load, test_size, configs)


def k_fold_cross_validation(models, enclosing_folder, datasets_to_load, num_splits, test_size, configs):
    """Performs k-fold cross validation on models. 80% of the images are used for training and 20% for testing.

    Args:
        models (list): The models to perform k-fold cross validation on.
        enclosing_folder (str): The name of the folder that contains the datasets to load. E.g. 'binary' or 'rating'.
        datasets_to_load (list[str]): A list containing the names of the datasets to be loaded.
        num_splits (int): The number of different ways in which the dataset will be split for k-fold cross
            validation.
        test_size (float): The proportion of images that will be used for testing in each fold. Ranges from 0-1. E.g.
            0.2 means 20% of images will be used for testing.
        configs (dict): Configuration settings used to set up the datasets for the Scikit-learn models and CNN models.
            Also includes training settings for CNN models.

    """
    for model in models:
        if issubclass(type(model), AbstractCNN):
            # The model is a CNN model
            # Set up datasets
            cnn_dataset_handler = CNNDatasetHandler(configs['cnn'])
            cnn_dataset_handler.create_dataset(enclosing_folder, datasets_to_load, print_info=True)
            X, y = cnn_dataset_handler.get_X_and_y()
            # K-Fold Cross Validation
            model.kfold_cross_validation(X, y, num_splits, test_size)
        elif issubclass(type(model), AbstractSklearn):
            # The model is a Scikit-learn model
            # Set up datasets
            sklearn_dataset_handler = SklearnDatasetHandler(configs['sklearn'])
            sklearn_dataset_handler.create_dataset(enclosing_folder, datasets_to_load, print_info=True)
            X, y = sklearn_dataset_handler.get_X_and_y()
            # K-Fold Cross Validation
            model.kfold_cross_validation(X, y, num_splits, test_size)
        else:
            # The model is not a Scikit-learn or CNN model
            raise ValueError("An invalid model was given as input. Each model must be a Scikit-learn or CNN model.")


def find_best_sklearn_seeds(models, enclosing_folder, datasets_to_load, test_size, configs, max_seed):
    """Tests Scikit-learn models on a variable number of seeds and prints the best seeds that were found.

    Args:
        models (list): The Scikit-learn models to find the best seeds for.
        enclosing_folder (str): The name of the folder that contains the datasets to load. E.g. 'binary' or 'rating'.
        datasets_to_load (list[str]): A list containing the names of the datasets to be loaded.
        test_size (float): The proportion of the dataset that will be used for testing. Ranges from 0-1.
            E.g. 0.2 means 20% of images will be used for testing.
        configs (dict): Configuration settings used to set up the datasets for the Scikit-learn models and CNN models.
            Also includes training settings for CNN models.
        max_seed (int): The number of seeds to be tested. E.g. 100 tests up to seed 100.

    """
    # Setup datasets
    sklearn_dataset_handler = SklearnDatasetHandler(configs['sklearn'])
    sklearn_dataset_handler.create_dataset(enclosing_folder, datasets_to_load, print_info=True)

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


def test_saved_basic_models():
    """Loads the models that were trained on images from a single surfing location (Bantham beach) and tests their
    performance on images from other locations."""

    # Configuration settings for the models
    configs = {
        'cnn': {
            'image_height': 40,
            'image_width': 40,
            'color_mode': 'rgb',
            'batch_size': 16,
            'epochs': 10
        },
        'sklearn': {
            'image_height': 128,
            'image_width': 128,
            'color_mode': 'rgb'
        }
    }

    # Load the models
    loaded_linear_cnn = LoadedCNN(configs['cnn'])
    loaded_linear_cnn.load_model('basic_linear_cnn')
    loaded_non_linear_cnn = LoadedCNN(configs['cnn'])
    loaded_non_linear_cnn.load_model('basic_non_linear_cnn')
    loaded_svm = LoadedSklearn()
    loaded_svm.load_model('basic_svm')
    loaded_rf = LoadedSklearn()
    loaded_rf.load_model('basic_rf')
    loaded_knn = LoadedSklearn()
    loaded_knn.load_model('basic_knn')

    # Setup datasets for the loaded CNN models and test them
    cnn_dataset_handler = CNNDatasetHandler(configs['cnn'])
    cnn_dataset_handler.create_dataset('binary', ['bantham'])
    print('\nPerformance on bantham dataset (the dataset the models were trained on):')
    X_train, X_val, X_test, y_train, y_val, y_test = cnn_dataset_handler.train_test_split(0.2, 3, 1)
    loaded_linear_cnn.test_model(X_test, y_test)
    X_train, X_val, X_test, y_train, y_val, y_test = cnn_dataset_handler.train_test_split(0.2, 1, 1)
    loaded_non_linear_cnn.test_model(X_test, y_test)

    # Setup datasets for the loaded Scikit-learn models
    sklearn_dataset_handler = SklearnDatasetHandler(configs['sklearn'])
    sklearn_dataset_handler.create_dataset('binary', ['bantham'])
    X_train, X_test, y_train, y_test = sklearn_dataset_handler.train_test_split(0.2, 3)

    # Test the loaded Scikit-learn models on the dataset that they were trained on
    loaded_svm.test_model(X_test, y_test)
    loaded_rf.test_model(X_test, y_test)
    loaded_knn.test_model(X_test, y_test)

    # Test the models on the datasets that they have not been trained on
    datasets_to_load = ['polzeath', 'porthowan', 'praa_sands', 'widemouth_bay']
    for dataset_name in datasets_to_load:
        print("\n-----------------------------------------")
        print(f'\nPerformance on {dataset_name} dataset:')

        # Set up dataset for CNN model
        cnn_dataset_handler.create_dataset('binary', [dataset_name])

        # Get the images and labels
        X_test, y_test = cnn_dataset_handler.get_X_and_y()

        # Test CNN models
        loaded_linear_cnn.test_model(X_test, y_test)
        loaded_non_linear_cnn.test_model(X_test, y_test)

        # Set up dataset for Scikit-learn models
        sklearn_dataset_handler.create_dataset('binary', [dataset_name])
        X_test, y_test = sklearn_dataset_handler.get_X_and_y()

        # Test Scikit-learn models
        loaded_svm.test_model(X_test, y_test)
        loaded_rf.test_model(X_test, y_test)
        loaded_knn.test_model(X_test, y_test)
