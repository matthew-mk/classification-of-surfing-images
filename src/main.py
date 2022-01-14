from classes.cnn import *
from classes.dataset_handler import *
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

IMAGES_DIR_1 = '../datasets/binary_1'
CATEGORIES = ['unsurfable', 'surfable']
TEST_SIZE = 0.2
TRAIN_SEED = 3
VAL_SEED = 1
CONFIG = {
    'image_height': 40,
    'image_width': 40,
    'color_mode': 'rgb',
    'batch_size': 16,
    'epochs': 50
}


def cnn_test():
    cnn_dataset_handler = CNNDatasetHandler(CONFIG)
    print('Loading dataset...')
    cnn_dataset_handler.load_dataset(IMAGES_DIR_1, CATEGORIES)
    print('Dataset loaded')
    X_train, X_val, X_test, y_train, y_val, y_test = cnn_dataset_handler.train_test_split(TEST_SIZE, TRAIN_SEED,
                                                                                          VAL_SEED)
    cnn = BasicCNN(CONFIG)
    cnn.train_model(X_train, y_train, X_val, y_val)
    cnn.plot_training_loss()
    cnn.plot_training_accuracy()
    cnn.test_model(X_test, y_test)


def sklearn_test():
    sklearn_dataset_handler = SKLearnDatasetHandler(CONFIG)
    print('Loading dataset...')
    sklearn_dataset_handler.load_dataset(IMAGES_DIR_1, CATEGORIES)
    print('Dataset loaded')
    X_train, X_test, y_train, y_test = sklearn_dataset_handler.train_test_split(TEST_SIZE, TRAIN_SEED)
    svm_model = SVC(C=4)
    rf_model = RandomForestClassifier()
    knn_model = KNeighborsClassifier(n_neighbors=1)
    svm_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)
    knn_model.fit(X_train, y_train)
    test_model(svm_model, False, X_test, y_test)
    test_model(rf_model, False, X_test, y_test)
    test_model(knn_model, False, X_test, y_test)


def main():
    sklearn_test()


if __name__ == '__main__':
    main()
