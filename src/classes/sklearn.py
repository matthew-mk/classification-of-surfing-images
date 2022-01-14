from utils import *


class SKLearn:
    """

    """
    def __init__(self, config, model):
        """

        """
        self.image_height = config['image_height']
        self.image_width = config['image_width']
        self.image_size = (self.image_height, self.image_width)
        self.color_mode = config['color_mode']
        self.num_channels = get_channels(self.color_mode)
        self.model = self.create_model()

    def create_model(self):
        """

        """
        raise NotImplementedError
