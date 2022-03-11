"""This module trains models to classify images of surfing locations based on whether the conditions in the images are
suitable for surfing or not (binary classification). The models can be trained using images from 1 to 5 surfing
locations. The datasets that are used in this module contain images from Surfline. """

import keras.losses
from utils.train_test_utils import *

ENCLOSING_FOLDER = 'binary'
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
        'epochs': 50
    },
    'sklearn': {
        'image_height': 128,
        'image_width': 128,
        'color_mode': 'rgb'
    }
}
SEEDS = {
    'linear_cnn': get_seed([0, 0, 0, 0, 0], NUM_DATASETS_TO_LOAD),
    'non_linear_cnn': get_seed([0, 0, 0, 0, 0], NUM_DATASETS_TO_LOAD),
    'svm': get_seed([14, 0, 5, 42, 32], NUM_DATASETS_TO_LOAD),
    'rf': get_seed([43, 0, 0, 50, 23], NUM_DATASETS_TO_LOAD),
    'knn': get_seed([50, 0, 8, 47, 43], NUM_DATASETS_TO_LOAD)
}

def main():
    # Create a list specifying the datasets that will be loaded, e.g. ['pipeline', 'lorne_point']
    DATASETS_TO_LOAD = DATASETS[:NUM_DATASETS_TO_LOAD]

    # Create instances of the models
    linear_cnn = LinearCNN(CONFIGS['cnn'])
    linear_cnn.compile_model(keras.optimizers.Adam(), keras.losses.BinaryCrossentropy(from_logits=True))
    non_linear_cnn = NonLinearCNN(CONFIGS['cnn'])
    non_linear_cnn.compile_model(keras.optimizers.Adam(), keras.losses.SparseCategoricalCrossentropy())
    svm = SVM()
    rf = RF()
    knn = BinaryKNN()

    # Train a single model
    train_and_test_model(linear_cnn, SEEDS['linear_cnn'], ENCLOSING_FOLDER, DATASETS_TO_LOAD, TEST_SIZE, CONFIGS)

    # Train multiple models
    # models_and_seeds = [(non_linear_cnn, SEEDS['non_linear_cnn']), (svm, SEEDS['svm']), (rf, SEEDS['rf']),
    #                     (knn, SEEDS['knn'])]
    # train_and_test_models(models_and_seeds, ENCLOSING_FOLDER, DATASETS_TO_LOAD, TEST_SIZE, CONFIGS)

    # Apply k-fold cross validation to one or more models
    # k_fold_cross_validation([non_linear_cnn], ENCLOSING_FOLDER, DATASETS_TO_LOAD, K_FOLD_SPLITS, TEST_SIZE, CONFIGS)

    # Test the Scikit-learn models on a specified number of seeds and find the seed that each model performed best on
    # find_best_sklearn_seeds([svm, rf, knn], ENCLOSING_FOLDER, DATASETS_TO_LOAD, TEST_SIZE, CONFIGS, 50)

if __name__ == '__main__':
    main()
