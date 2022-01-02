from utils import *
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split

BASIC_IMAGES_DIR = 'images/basic'
CATEGORIES = ['unsurfable', 'surfable']
COLOR_MODE = 'rgb'
NUM_CHANNELS = get_channels(COLOR_MODE)
IMAGE_HEIGHT = 40
IMAGE_WIDTH = 40
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)
BATCH_SIZE = 16
EPOCHS = 300

# Set up the images array (X) and labels array (y)
X, y = extract_images_and_labels(BASIC_IMAGES_DIR, CATEGORIES, COLOR_MODE, IMAGE_SIZE)
X, y = np.array(X), np.array(y)
X = X.reshape(len(X), IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS)

# Set up the training, validation, and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal", input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

model = keras.Sequential([
    data_augmentation,
    layers.Input((IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS)),
    layers.Conv2D(16, 3, padding='same'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding='same'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding='same'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128),
    layers.Dense(64),
    layers.Dense(1)
])

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    'models/basic_cnn2.h5',
    monitor='val_loss',
    mode='min',
    save_best_only=True
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[model_checkpoint_callback]
)

plot_cnn_loss(history, EPOCHS)
plot_cnn_accuracy(history, EPOCHS)
cnn_model_accuracy_and_loss(model, X_val, y_val, X_test, y_test)

# loaded_model = keras.models.load_model('models/basic_cnn.h5')
# cnn_model_accuracy_and_loss(loaded_model, X_val, y_val, X_test, y_test)

# loaded_model.summary()
