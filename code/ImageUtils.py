import numpy as np
import random
import matplotlib.pyplot as plt
import os

def parse_record(record, augment):
    """ Parse a record to an image and perform data preprocessing.

    Args:
        record: An array of shape [3072,]. One row of the x_* matrix.
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [3, 32, 32].
    """
    # Reshape from [depth * height * width] to [depth, height, width].
    depth_major = record.reshape((3, 32, 32))

    # Convert from [depth, height, width] to [height, width, depth]
    image = np.transpose(depth_major, [1, 2, 0])

    image = preprocess_image(image, augment)

    # Convert from [height, width, depth] to [depth, height, width]
    image = np.transpose(image, [2, 0, 1])

    return image

def preprocess_image(image, augment):
    """ Preprocess a single image of shape [height, width, depth].

    Args:
        image: An array of shape [32, 32, 3].
        training: A boolean. Determine whether it is in training mode.
    
    Returns:
        image: An array of shape [32, 32, 3].
    """
    if augment:
        # Resize the image to add four extra pixels on each side.
        frame = np.full((40, 40, 3), 255.0)
        frame[4:36, 4:36, :] = image

        # Randomly crop a 32x32 image
        x = random.randint(0, 8)
        y = random.randint(0, 8)
        image = frame[x : x + 32, y : y + 32, :]
        assert image.shape == (32, 32, 3)

        # Randomly flip the image horizontally.
        if random.randint(0, 1) == 1:
            flipped = np.empty(image.shape)
            for y in range(0, 32):
                flipped[:, y, :] = image[:, 31 - y, :]
            image = flipped

    assert(image.shape[2] == 3)
    # Normalize
    for channel in range(3):
        image[:, :, channel] = (image[:, :, channel] - np.mean(image[:, :, channel])) / np.std(image[:, :, channel])

    return image

def show_image(image, predicted_label, actual_label=""):
    image = np.transpose(image.reshape(3, 32, 32), [1,2,0])
    plt.imshow(image.astype('uint8'))
    plt.title("predicted: {}, actual: {}".format(predicted_label, actual_label))
    plt.show()