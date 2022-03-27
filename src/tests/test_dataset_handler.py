"""This module contains unit tests for the DatasetHandler classes. """

import unittest
from classes.dataset_handler import *
from classes.sklearn import *
from classes.cnn import *


class TestDatasetHandler(unittest.TestCase):

    def assert_dataset_size(self, dataset_handler, enclosing_folder, location, num_images):
        dataset_handler.create_dataset(enclosing_folder, [location])
        X, y = dataset_handler.get_X(), dataset_handler.get_y()
        self.assertEqual(len(X), num_images)
        self.assertEqual(len(y), num_images)

    def test_sklearn_create_dataset_single_location(self):
        dataset_handler = SklearnDatasetHandler({'image_height': 128,
                                                 'image_width': 128,
                                                 'color_mode': 'rgb'})
        self.assert_dataset_size(dataset_handler, 'binary', 'bantham', 300)
        self.assert_dataset_size(dataset_handler, 'binary', 'polzeath', 100)
        self.assert_dataset_size(dataset_handler, 'binary', 'porthowan', 100)
        self.assert_dataset_size(dataset_handler, 'binary', 'praa_sands', 100)
        self.assert_dataset_size(dataset_handler, 'binary', 'widemouth_bay', 100)
        self.assert_dataset_size(dataset_handler, 'binary', 'pipeline', 300)
        self.assert_dataset_size(dataset_handler, 'binary', 'lorne_point', 100)
        self.assert_dataset_size(dataset_handler, 'binary', 'noosa_heads', 100)
        self.assert_dataset_size(dataset_handler, 'binary', 'marias_beachfront', 100)
        self.assert_dataset_size(dataset_handler, 'binary', 'rocky_point', 100)
        self.assert_dataset_size(dataset_handler, 'rating', 'bantham', 300)
        self.assert_dataset_size(dataset_handler, 'rating', 'polzeath', 100)
        self.assert_dataset_size(dataset_handler, 'rating', 'porthowan', 100)
        self.assert_dataset_size(dataset_handler, 'rating', 'praa_sands', 100)
        self.assert_dataset_size(dataset_handler, 'rating', 'widemouth_bay', 100)
        self.assert_dataset_size(dataset_handler, 'rating', 'pipeline', 300)
        self.assert_dataset_size(dataset_handler, 'rating', 'lorne_point', 100)
        self.assert_dataset_size(dataset_handler, 'rating', 'noosa_heads', 100)
        self.assert_dataset_size(dataset_handler, 'rating', 'marias_beachfront', 100)
        self.assert_dataset_size(dataset_handler, 'rating', 'rocky_point', 100)

    def test_cnn_create_dataset_single_location(self):
        dataset_handler = CNNDatasetHandler({'image_height': 40,
                                             'image_width': 40,
                                             'color_mode': 'rgb'})
        self.assert_dataset_size(dataset_handler, 'binary', 'bantham', 300)
        self.assert_dataset_size(dataset_handler, 'binary', 'polzeath', 100)
        self.assert_dataset_size(dataset_handler, 'binary', 'porthowan', 100)
        self.assert_dataset_size(dataset_handler, 'binary', 'praa_sands', 100)
        self.assert_dataset_size(dataset_handler, 'binary', 'widemouth_bay', 100)
        self.assert_dataset_size(dataset_handler, 'binary', 'pipeline', 300)
        self.assert_dataset_size(dataset_handler, 'binary', 'lorne_point', 100)
        self.assert_dataset_size(dataset_handler, 'binary', 'noosa_heads', 100)
        self.assert_dataset_size(dataset_handler, 'binary', 'marias_beachfront', 100)
        self.assert_dataset_size(dataset_handler, 'binary', 'rocky_point', 100)
        self.assert_dataset_size(dataset_handler, 'rating', 'bantham', 300)
        self.assert_dataset_size(dataset_handler, 'rating', 'polzeath', 100)
        self.assert_dataset_size(dataset_handler, 'rating', 'porthowan', 100)
        self.assert_dataset_size(dataset_handler, 'rating', 'praa_sands', 100)
        self.assert_dataset_size(dataset_handler, 'rating', 'widemouth_bay', 100)
        self.assert_dataset_size(dataset_handler, 'rating', 'pipeline', 300)
        self.assert_dataset_size(dataset_handler, 'rating', 'lorne_point', 100)
        self.assert_dataset_size(dataset_handler, 'rating', 'noosa_heads', 100)
        self.assert_dataset_size(dataset_handler, 'rating', 'marias_beachfront', 100)
        self.assert_dataset_size(dataset_handler, 'rating', 'rocky_point', 100)

    def test_sklearn_create_dataset_merged_locations(self):
        dataset_handler = SklearnDatasetHandler({'image_height': 128,
                                                 'image_width': 128,
                                                 'color_mode': 'rgb'})
        dataset_handler.create_dataset('binary', ['bantham','polzeath','porthowan','praa_sands','widemouth_bay'])
        X, y = dataset_handler.get_X(), dataset_handler.get_y()
        self.assertEqual(len(X), 700)
        self.assertEqual(len(y), 700)
        dataset_handler.create_dataset('rating', ['bantham','polzeath','porthowan','praa_sands','widemouth_bay'])
        X, y = dataset_handler.get_X(), dataset_handler.get_y()
        self.assertEqual(len(X), 700)
        self.assertEqual(len(y), 700)
        dataset_handler.create_dataset('binary', ['pipeline','lorne_point','noosa_heads','marias_beachfront','rocky_point'])
        X, y = dataset_handler.get_X(), dataset_handler.get_y()
        self.assertEqual(len(X), 700)
        self.assertEqual(len(y), 700)
        dataset_handler.create_dataset('rating', ['pipeline','lorne_point','noosa_heads','marias_beachfront','rocky_point'])
        X, y = dataset_handler.get_X(), dataset_handler.get_y()
        self.assertEqual(len(X), 700)
        self.assertEqual(len(y), 700)

    def test_cnn_create_dataset_merged_locations(self):
        dataset_handler = CNNDatasetHandler({'image_height': 40,
                                             'image_width': 40,
                                             'color_mode': 'rgb'})
        dataset_handler.create_dataset('binary', ['bantham','polzeath','porthowan','praa_sands','widemouth_bay'])
        X, y = dataset_handler.get_X(), dataset_handler.get_y()
        self.assertEqual(len(X), 700)
        self.assertEqual(len(y), 700)
        dataset_handler.create_dataset('rating', ['bantham','polzeath','porthowan','praa_sands','widemouth_bay'])
        X, y = dataset_handler.get_X(), dataset_handler.get_y()
        self.assertEqual(len(X), 700)
        self.assertEqual(len(y), 700)
        dataset_handler.create_dataset('binary', ['pipeline','lorne_point','noosa_heads','marias_beachfront','rocky_point'])
        X, y = dataset_handler.get_X(), dataset_handler.get_y()
        self.assertEqual(len(X), 700)
        self.assertEqual(len(y), 700)
        dataset_handler.create_dataset('rating', ['pipeline','lorne_point','noosa_heads','marias_beachfront','rocky_point'])
        X, y = dataset_handler.get_X(), dataset_handler.get_y()
        self.assertEqual(len(X), 700)
        self.assertEqual(len(y), 700)

    def test_sklearn_create_dataset_raises(self):
        dataset_handler = SklearnDatasetHandler({'image_height': 128,
                                                 'image_width': 128,
                                                 'color_mode': 'rgb'})
        with self.assertRaises(FileNotFoundError):
            dataset_handler.create_dataset('binary', ['unknown'])
        with self.assertRaises(FileNotFoundError):
            dataset_handler.create_dataset('rating', ['unknown'])

    def test_cnn_create_dataset_raises(self):
        dataset_handler = CNNDatasetHandler({'image_height': 40,
                                             'image_width': 40,
                                             'color_mode': 'rgb'})
        with self.assertRaises(FileNotFoundError):
            dataset_handler.create_dataset('binary', ['unknown'])
        with self.assertRaises(FileNotFoundError):
            dataset_handler.create_dataset('rating', ['unknown'])

    def test_categories(self):
        self.assertEqual(AbstractDatasetHandler.get_categories('binary'), ['unsurfable', 'surfable'])
        self.assertEqual(AbstractDatasetHandler.get_categories('rating'), ['1', '2', '3', '4', '5'])
        with self.assertRaises(ValueError):
            AbstractDatasetHandler.get_categories('invalid_folder')
