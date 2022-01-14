from basic_train import *
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

BASIC_IMAGES_DIR = 'datasets/binary_1'
CATEGORIES = ['unsurfable', 'surfable']
COLOR_MODE = 'rgb'
NUM_CHANNELS = get_channels(COLOR_MODE)
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)

X, y = extract_images_and_labels(BASIC_IMAGES_DIR, CATEGORIES, COLOR_MODE, IMAGE_SIZE)
models = [('SVM', SVC(C=4)), ('RF', RandomForestClassifier()), ('KNN', KNeighborsClassifier(n_neighbors=1))]
kfold_cross_validation(models, X, y, 10)

# Plot a confusion matrix
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
# y_predicted = svm_model.predict(X_test)
# plot_confusion_matrix(y_test, y_predicted)

# Save a model
# pickle.dump(model, open('saved_models/basic_svm1.sav', 'wb'))

# Load a model
# loaded_model = pickle.load(open('saved_models/basic_svm.sav', 'rb'))
# model_test_accuracy(loaded_model, False, X_test, y_test)
