"""This module trains models to classify images based on a rating of how good the conditions are (multi-class
classification). The models are trained using images from multiple surfing locations and evaluated using appropriate
metrics. The datasets that are used in this module contain images from Surfline. """

import keras.losses
from classes.cnn import *
from classes.sklearn import *
from classes.dataset_handler import *
from utils.train_test_utils import *

ENCLOSING_FOLDER = 'rating'
DATASETS = ['pipeline', 'lorne_point', 'noosa_heads', 'marias_beachfront', 'rocky_point']
NUM_DATASETS_TO_LOAD = 1
TEST_SIZE = 0.2
K_FOLD_SPLITS = 5
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
SEEDS = {
    'cnn': get_seed([1, 0, 0, 0, 0], NUM_DATASETS_TO_LOAD),
    'svm': get_seed([25, 16, 20, 11, 27], NUM_DATASETS_TO_LOAD),
    'rf': get_seed([34, 2, 20, 26, 27], NUM_DATASETS_TO_LOAD),
    'knn': get_seed([43, 5, 1, 36, 0], NUM_DATASETS_TO_LOAD)
}

def main():
    # Create a list specifying the datasets that will be loaded, e.g. ['pipeline', 'lorne_point']
    DATASETS_TO_LOAD = DATASETS[:NUM_DATASETS_TO_LOAD]

    # Create instances of the models
    cnn = RatingCNN(CONFIGS['cnn'])
    cnn.compile_model(keras.optimizers.Adam(), keras.losses.SparseCategoricalCrossentropy())
    svm = SVM()
    rf = RF()
    knn = RatingKNN()

    # Train a single model
    train_and_test_model(cnn, SEEDS['cnn'], ENCLOSING_FOLDER, DATASETS_TO_LOAD, TEST_SIZE, CONFIGS)

    # Train multiple models
    # models_and_seeds = [(svm, SEEDS['svm']), (rf, SEEDS['rf']), (knn, SEEDS['knn'])]
    # train_and_test_models(models_and_seeds, ENCLOSING_FOLDER, DATASETS_TO_LOAD, TEST_SIZE, CONFIGS)

    # Apply k-fold cross validation to one or more models
    # k_fold_cross_validation([cnn], ENCLOSING_FOLDER, DATASETS_TO_LOAD, K_FOLD_SPLITS, TEST_SIZE, CONFIGS)

    # Test the Scikit-learn models on a specified number of seeds and find the seed that each model performed best on
    # find_best_sklearn_seeds([svm, rf, knn], ENCLOSING_FOLDER, DATASETS_TO_LOAD, TEST_SIZE, CONFIGS, 50)

if __name__ == '__main__':
    main()

