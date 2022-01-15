from classes.cnn import *
from classes.sklearn import *
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
    cnn_dataset_handler.load_dataset(IMAGES_DIR_1, CATEGORIES)
    X_train, X_val, X_test, y_train, y_val, y_test = cnn_dataset_handler.train_test_split(TEST_SIZE, TRAIN_SEED,
                                                                                          VAL_SEED)
    cnn = BasicCNN(CONFIG)
    cnn.train_model(X_train, y_train, X_val, y_val)
    cnn.plot_training_loss()
    cnn.plot_training_accuracy()
    cnn.test_model(X_test, y_test)


def sklearn_test():
    sklearn_dataset_handler = SKLearnDatasetHandler(CONFIG)
    sklearn_dataset_handler.load_dataset(IMAGES_DIR_1, CATEGORIES)
    X_train, X_test, y_train, y_test = sklearn_dataset_handler.train_test_split(TEST_SIZE, TRAIN_SEED)

    svm = SKLearn(SVC(C=4))
    rf = SKLearn(RandomForestClassifier())
    knn = SKLearn(KNeighborsClassifier(n_neighbors=1))

    svm.train_model(X_train, y_train)
    rf.train_model(X_train, y_train)
    knn.train_model(X_train, y_train)

    print('SVM Model')
    svm.test_model(X_test, y_test)
    print('RF Model')
    rf.test_model(X_test, y_test)
    print('KNN Model')
    knn.test_model(X_test, y_test)


def main():
    sklearn_test()


if __name__ == '__main__':
    main()
