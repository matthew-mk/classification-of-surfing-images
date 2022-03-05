"""This module trains models to classify images of surfing locations based on whether the conditions in the images are
suitable for surfing or not (binary classification). The models can be trained using images from 1 to 5 surfing
locations."""

import keras.losses
from utils.train_test_utils import *

ENCLOSING_FOLDER = 'binary'
MAGICSEAWEED_DATASETS = ['bantham', 'polzeath', 'porthowan', 'praa_sands', 'widemouth_bay']
NUM_DATASETS_TO_LOAD = 5
TEST_SIZE = 0.2
K_FOLD_SPLITS = 3
CONFIGS = {
    'cnn': {
        'image_height': 40,
        'image_width': 40,
        'color_mode': 'rgb',
        'batch_size': 16,
        'epochs': 3
    },
    'sklearn': {
        'image_height': 128,
        'image_width': 128,
        'color_mode': 'rgb'
    }
}
BEST_SEEDS = {
    'cnn': get_seed([3, 0, 0, 0, 123], NUM_DATASETS_TO_LOAD),
    'svm': get_seed([3, 88, 3, 47, 8], NUM_DATASETS_TO_LOAD),
    'rf': get_seed([3, 72, 30, 13, 73], NUM_DATASETS_TO_LOAD),
    'knn': get_seed([3, 50, 1, 26, 73], NUM_DATASETS_TO_LOAD)
}

def main():
    # Get the datasets to load based on the number of locations that was specified above
    DATASETS_TO_LOAD = MAGICSEAWEED_DATASETS[:NUM_DATASETS_TO_LOAD]

    # Create instances of the models
    linear_cnn = LinearCNN(CONFIGS['cnn'])
    linear_cnn.compile_model(keras.optimizers.Adam(), keras.losses.BinaryCrossentropy(from_logits=True))
    nonlinear_cnn = NonLinearCNN(CONFIGS['cnn'])
    nonlinear_cnn.compile_model(keras.optimizers.Adam(), keras.losses.SparseCategoricalCrossentropy())
    svm = SVM()
    rf = RF()
    knn = KNN()

    # Train a single model
    train_and_test_model(linear_cnn, BEST_SEEDS['cnn'], ENCLOSING_FOLDER, DATASETS_TO_LOAD, TEST_SIZE, CONFIGS)

    # Train multiple models
    # models_and_seeds = [(linear_cnn, BEST_SEEDS['cnn']), (svm, BEST_SEEDS['svm']), (rf, BEST_SEEDS['rf']),
    #                     (knn, BEST_SEEDS['knn'])]
    # train_and_test_models(models_and_seeds, ENCLOSING_FOLDER, DATASETS_TO_LOAD, TEST_SIZE, CONFIGS)

    # Apply k-fold cross validation to one or more models
    # k_fold_cross_validation([linear_cnn], ENCLOSING_FOLDER, DATASETS_TO_LOAD, K_FOLD_SPLITS, TEST_SIZE, CONFIGS)

    # Test the models that have been saved that were trained on images from a single surfing location (Bantham beach)
    # test_saved_basic_models()

if __name__ == '__main__':
    main()
