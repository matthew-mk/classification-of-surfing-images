from utils import *
from tensorflow import keras
from sklearn.model_selection import train_test_split
import os


class DatasetHandler:
    """

    """
    def __init__(self, config):
        """

        """
        self.image_height = config['image_height']
        self.image_width = config['image_width']
        self.image_size = (self.image_height, self.image_width)
        self.color_mode = config['color_mode']
        self.num_channels = get_channels(self.color_mode)
        self.X, self.y = [], []
        self.X_train, self.y_train = [], []
        self.X_test, self.y_test = [], []

    def load_dataset(self, path_to_dir, categories):
        """Loads the images from a dataset, assigns a label to each image, and does some preprocessing.

        The names of the folders that are specified in the categories array are looped through inside of the
        directory, with each folder containing the images for a particular class. Each image is assigned a label (the
        index of the name of the folder in the categories array) and converted to be the size and color mode that are
        defined by this class. In addition, each image is transformed into a numpy array containing the image's pixel
        values and the pixel values are normalized.

        Args:
            path_to_dir (str): The path to the dataset.
            categories (list[str]): An array containing the names of the folders in the directory that have the images.
                Each folder should contain the images for a separate class. The index of each item in the array
                represents the label that category will be given.

        """
        self.X = []
        self.y = []
        for category in categories:
            category_path = os.path.join(path_to_dir, category)
            category_label = categories.index(category)
            for file in os.listdir(category_path):
                if file.endswith('.png'):
                    img_path = os.path.join(category_path, file)
                    img = keras.preprocessing.image.load_img(img_path,
                                                             color_mode=self.color_mode,
                                                             target_size=self.image_size)
                    img_array = np.array(img).flatten()
                    img_array = img_array / 255  # normalize the data
                    self.X.append(img_array)
                    self.y.append(category_label)

    def train_test_split(self):
        """Splits the dataset (X and y) into datasets that can be used for training and testing.

        Raises:
            NotImplementedError: Subclasses must implement this method.

        """
        raise NotImplementedError

    def get_X(self):
        """Returns the list of images, where each image is represented as a pixel array.

        Returns:
            X (list[int]): The list of images.

        """
        return self.X

    def get_y(self):
        """Returns the list of labels.

        Returns:
            y (list[int]): The labels of the images.

        """
        return self.y

    def get_X_and_y(self):
        """Returns the list of images and the list of corresponding labels.

        Returns:
            X (list[int]): The list of images.
            y (list[int]): The labels of the images.

        """
        return self.X, self.y

    def get_image_height(self):
        """Returns the image height that each of the images in the dataset are converted to have.

        Returns:
            image_height (int): The image height.

        """
        return self.image_height

    def get_image_width(self):
        """Returns the image width that each of the images in the dataset are converted to have.

        Returns:
            image_width (int): The image width.

        """
        return self.image_width

    def get_image_size(self):
        """Returns the image size that each of the images in the dataset are converted to have.

        Returns:
            image_size ((int, int)): The image size.

        """
        return self.image_size

    def get_color_mode(self):
        """Returns the color mode that each of the images in the dataset are converted to have.

        Returns:
            color_mode (str): The color mode.

        """
        return self.color_mode


class CNNDatasetHandler(DatasetHandler):
    """

    """
    def __init__(self, config):
        """

        """
        super().__init__(config)
        self.X_val, self.y_val = [], []

    def train_test_split(self, test_size, train_test_seed, val_seed):
        """Splits the dataset (X and y) into datasets that can be used for training, validation and testing.

        Arguments:
            test_size (float): The proportion of the dataset that will be used for testing. Ranges from 0-1.
            train_test_seed (int): A reproducible shuffle of the training and testing data.
            val_seed (int): A reproducible shuffle of the validation data.

        Returns:
            X_train (numpy.ndarray): The images in the training dataset.
            X_val (numpy.ndarray): The images in the validation dataset.
            X_test (numpy.ndarray): The images in the test dataset.
            y_train (numpy.ndarray): The labels in the training dataset.
            y_val (numpy.ndarray): The labels in the validation dataset.
            y_test (numpy.ndarray): The labels in the test dataset.

        """
        self.X, self.y = np.array(self.X), np.array(self.y)
        self.X = self.X.reshape(len(self.X), self.image_height, self.image_width, self.num_channels)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                                                test_size=test_size,
                                                                                random_state=train_test_seed)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train,
                                                                              test_size=test_size,
                                                                              random_state=val_seed)
        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test


class SKLearnDatasetHandler(DatasetHandler):
    """

    """
    def __init__(self, config):
        """

        """
        super().__init__(config)

    def train_test_split(self, test_size, seed):
        """Splits the dataset (X and y) into datasets that can be used for training and testing.

        Arguments:
            test_size (float): The proportion of the dataset that will be used for testing. Ranges from 0 - 1.
            seed (int): A reproducible shuffle of the dataset.

        Returns:
            X_train (numpy.ndarray): The images in the training dataset.
            X_test (numpy.ndarray): The images in the test dataset.
            y_train (numpy.ndarray): The labels in the training dataset.
            y_test (numpy.ndarray): The labels in the test dataset.

        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size,
                                                                                random_state=seed)
        return self.X_train, self.X_test, self.y_train, self.y_test


