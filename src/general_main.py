from classes.cnn import *
from classes.sklearn import *
from classes.dataset_handler import *

"""

"""

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
