"""This module trains models to classify images of surfing locations based on whether the conditions in the images are
suitable for surfing or not (binary classification). The models can be trained using images from 1 to 5 surfing
locations."""

from utils.train_test_utils import *

DATASET_TYPE = 'binary'
CATEGORIES = ['unsurfable', 'surfable']
NUM_LOCATIONS = 1
TEST_SIZE = 0.2
K_FOLD_SPLITS = 3
NUM_SEEDS_TO_TEST = 10
DATASETS_TO_LOAD = BaseDatasetHandler.get_dataset_names(DATASET_TYPE, NUM_LOCATIONS)
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

def main():
    # Create instances of the models
    cnn = CNN(CONFIGS['cnn'])
    cnn.compile_model(keras.optimizers.Adam(), keras.losses.BinaryCrossentropy(from_logits=True))
    svm = SVM()
    rf = RF()
    knn = KNN()

    # Train a single model
    # train_and_test_model(svm, BEST_SEEDS['svm'], DATASETS_TO_LOAD, CATEGORIES, TEST_SIZE, CONFIGS)

    # Train multiple models
    models_and_seeds = [(svm, BEST_SEEDS['svm']), (rf, BEST_SEEDS['rf']), (knn, BEST_SEEDS['knn'])]
    train_and_test_models(models_and_seeds, DATASETS_TO_LOAD, CATEGORIES, TEST_SIZE, CONFIGS)

    # Apply k-fold cross validation to one or more models
    # k_fold_cross_validation([cnn, svm, rf, knn], DATASETS_TO_LOAD, CATEGORIES, K_FOLD_SPLITS, CONFIGS)

    # Test the models that have been saved that were trained on images from a single surfing location (Bantham beach)
    # test_saved_basic_models()

if __name__ == '__main__':
    main()
