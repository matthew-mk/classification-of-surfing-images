"""This module contains unit tests for saving and loading models. """

import unittest
from classes.dataset_handler import *
from classes.sklearn import *
from classes.cnn import *
import os


class TestSaveAndLoad(unittest.TestCase):

    def test_cnn_save_and_load(self):
        cnn_config = {'image_height': 40,
                      'image_width': 40,
                      'color_mode': 'rgb',
                      'batch_size': 16,
                      'epochs': 50}
        saved_cnn = BinaryCNN(cnn_config)
        saved_cnn.compile_model(keras.optimizers.Adam(), keras.losses.SparseCategoricalCrossentropy())
        saved_cnn.save_model('unit_test_model')
        loaded_cnn = LoadedCNN(cnn_config)
        loaded_cnn.load_model('unit_test_model')
        os.remove('../../saved_models/unit_test_model.h5')
        self.assertTrue(type(saved_cnn.model) is type(loaded_cnn.model))
        self.assertTrue(saved_cnn.model.get_config() == loaded_cnn.model.get_config())

    def test_sklearn_save_and_load(self):
        saved_svm = SVM()
        saved_svm.save_model('unit_test_model')
        loaded_svm = LoadedSklearn()
        loaded_svm.load_model('unit_test_model')
        os.remove('../../saved_models/unit_test_model.sav')
        self.assertTrue(type(saved_svm.model) is type(loaded_svm.model))

    def test_save_raises(self):
        with self.assertRaises(ValueError):
            model = BinaryCNN({'image_height': 40,
                               'image_width': 40,
                               'color_mode': 'rgb',
                               'batch_size': 16,
                               'epochs': 50})
            model.compile_model(keras.optimizers.Adam(), keras.losses.SparseCategoricalCrossentropy())
            model.save_model(' ')
        with self.assertRaises(ValueError):
            model = SVM()
            model.save_model(' ')
