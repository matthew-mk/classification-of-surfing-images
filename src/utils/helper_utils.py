"""This module contains utility functions that help with various areas of development."""


def get_channels(color_mode):
    """Returns the number of channels that a color mode has.

    Args:
        color_mode (str): The color mode. Either 'grayscale', 'rgb', or 'rgba'.

    Returns:
        int: The number of channels.
    """
    if color_mode == 'grayscale':
        return 1
    elif color_mode == 'rgb':
        return 3
    else:
        return 4


def get_seed(seeds, num_locations):
    """Returns the seed at a particular index in a list of seeds.

    E.g. if num_locations is 4, the 4th seed in the list will be returned.

    Args:
        seeds (list[int]): A list of seeds.
        num_locations (int): The number of locations that images are being loaded from.

    Returns:
        seed (int): A seed.

    """
    if num_locations <= 0:
        raise ValueError("The number of locations must be greater than 0.")
    if num_locations > len(seeds):
        raise ValueError(f"The number of locations is too high. Maximum = {len(seeds)}")

    return seeds[num_locations - 1]
