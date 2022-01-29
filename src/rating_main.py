"""This module trains models to classify images based on a rating of how good the conditions are
(multi-class classification). The models are trained using images from multiple surfing locations and evaluated using
appropriate metrics."""

from classes.cnn import *
from classes.sklearn import *
from classes.dataset_handler import *

CATEGORIES = ['0', '1', '2', '3', '4', '5']
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
