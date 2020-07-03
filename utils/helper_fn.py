from matplotlib import  pyplot as plt
import tensorflow as tf
import numpy as np
import re
'''
Helper functions to display/visualize data
'''

# Helper function to read files in numerical order
numbers = re.compile(r'(\d+)')
def num_sort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])

    return parts


# Helper function to create mask
def create_mask(pred_mask):
    """

    :param pred_mask: raw predicted mask from the model of size 1x224x224x3 --> 3 channels, one per label prediction
    :return: processed mask of size 224x224 ---> passes on the maximum channel among the 3 channels.

    """

    pred_mask = tf.argmax(pred_mask, axis=-1)

    return pred_mask[0]


# Helper function to display data
def display(display_list):
    """

    :param display_list: list of arrays/images to display
    :return: None

    """

    fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(15,5))
    title = ['Image with Predicted Mask Contour', 'True Mask', 'Predicted Mask']
    ax1.imshow(display_list[0])
    ax1.set_title(title[0])
    ax1.axis('off')
    ax2.imshow(display_list[1])
    ax2.set_title(title[1])
    ax2.axis('off')
    ax3.imshow(display_list[2])
    ax3.set_title(title[2])
    ax1.contour(display_list[2], colors='r',linewidths=5,levels=[0.5])
    ax3.axis('off')
    plt.show()


def show_predictions(dataset=None, model=None, num=1):
    """

    :param dataset: iterator dataset to produce images and masks for visualization
    :param model: model needed to make predictions on provided data
    :param num: number of elements to visualize
    :return: None

    """

    for image, mask in dataset.take(num):
        pred_mask = model(image)
        display([np.squeeze(image[0]), np.squeeze(mask[0]), create_mask(pred_mask)])

def visualize_training(results):
    """

    :param results: training history
    :return: None

    """

    plt.figure(figsize=(10, 5))
    plt.title("Learning curve")
    plt.plot(results.history["loss"], label="loss")
    plt.plot(results.history["val_loss"], label="val_loss")
    plt.plot(np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("log_loss")
    plt.legend()
    plt.show()