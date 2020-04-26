from matplotlib import  pyplot as plt
import tensorflow as tf
import numpy as np
'''
Helper functions to display/visualize data
'''

# Helper function to create mask
def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


# Helper function to display data
def display(display_list):
    fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(15,5))
    title = ['Input Image with Predicted Mask', 'True Mask', 'Predicted Mask']
    ax1.imshow(tf.keras.preprocessing.image.array_to_img(display_list[0]))
    ax1.set_title(title[0])
    ax2.imshow(tf.keras.preprocessing.image.array_to_img(display_list[1]))
    ax2.set_title(title[1])
    ax3.imshow(tf.keras.preprocessing.image.array_to_img(display_list[2]))
    ax3.set_title(title[2])
    ax1.contour(tf.keras.preprocessing.image.array_to_img(display_list[2]), colors='r',linewidths=5,levels=[0.5])
    plt.axis('off')
    plt.show()
    # plt.figure(figsize=(15, 5))
    # for i in range(len(display_list)):
    #     plt.subplot(1, len(display_list), i+1)
    #     plt.title(title[i])
    #     plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    #     plt.axis('off')
    # plt.show()


def show_predictions(dataset=None, model=None, num=1):
    for image, mask in dataset.take(num):
        pred_mask = model(image)
        display([image[0], mask[0], create_mask(pred_mask)])


def visualize_training(results):
    plt.figure(figsize=(10, 5))
    plt.title("Learning curve")
    plt.plot(results.history["loss"], label="loss")
    plt.plot(results.history["val_loss"], label="val_loss")
    plt.plot(np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("log_loss")
    plt.legend()
    plt.show()