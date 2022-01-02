from utils import *
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pickle

BASIC_IMAGES_DIR = 'images/basic'
CATEGORIES = ['unsurfable', 'surfable']
COLOR_MODE = 'rgb'
NUM_CHANNELS = get_channels(COLOR_MODE)
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)

X, y = extract_images_and_labels(BASIC_IMAGES_DIR, CATEGORIES, COLOR_MODE, IMAGE_SIZE)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

svm_model = SVC(C=4)
rf_model = RandomForestClassifier()
knn_model = KNeighborsClassifier(n_neighbors=1)

models = [('SVM Model', svm_model), ('RF Model', rf_model), ('KNN Model', knn_model)]

print(type(svm_model))

for name, model in models:
    print(name)
    model.fit(X_train, y_train)
    model_test_accuracy(model, X_test, y_test)
    print()

# Plot a confusion matrix
# y_predicted = svm_model.predict(X_test)
# plot_confusion_matrix(y_test, y_predicted)

# Save a model
# pickle.dump(model, open('models/basic_svm1.sav', 'wb'))

# Load a model
# loaded_model = pickle.load(open('models/basic_svm.sav', 'rb'))
# model_test_accuracy(loaded_model, X_test, y_test)
