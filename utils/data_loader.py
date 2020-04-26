import os
import numpy as np
from matplotlib import pyplot as plt
from termcolor import colored
from sklearn.model_selection import train_test_split
import random
import tensorflow as tf
sep = os.sep

'''
This module handled loading and pre-processing the data
1- Data is loaded from disk and verified
2- Data size is processe to match model architecture
3- Data is split into train and validation
4- Iterator based on DATASET API is built
'''
###############################################


def data_loader(batch_size=2, buffer_size=2, visualize=True, masks_path = 'data_files'+sep+'masks', images_path = 'data_files'+sep+'images'):

    # Load images and masks from disk...
    ls_masks = os.listdir(masks_path)
    ls_images = os.listdir(images_path)
    masks = []
    images = []
    matching = 0
    for i, mask in enumerate(ls_masks):
        np_mask = np.load(os.path.join(masks_path,mask))
        image_np = plt.imread(os.path.join(images_path,ls_images[i]))
        if mask.split('.')[0] == ls_images[i].split('.')[0]:
            matching +=1
            masks.append(np_mask)
            images.append(image_np)

    matched = matching
    unmatched = len(images) - matching
    print(colored('[MATCHED]:','green'), '{0:} elements of out {1:}'.format(matched, len(images)))
    print(colored('[UNMATCHED]:','red'), '{0:} elements out of {1:}'.format(unmatched, len(images)))




    '''
    u-net model expects images with 3-channels thus we need
    to exapnd the channels for the grayscale to 3 channels
    Additionally, the model expects a 4-dim tensor 
    '''
    images = np.asarray(images)
    masks = np.array(masks)
    images = np.repeat(images[..., np.newaxis], 3, -1) # Convert to three channels
    masks = np.expand_dims(masks, axis=-1) # expand dimension of gray-scale



    # Split the data into train and validation 75%, 25% split.
    X_train, X_valid, y_train, y_valid = train_test_split(images, masks, test_size=0.25, random_state=42)



    if visualize:
        ix = random.randint(0, len(X_train)-1)
        has_mask = y_train[ix].max() > 0  # mask indicator

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        ax1.imshow(X_train[ix, ..., 0],  interpolation='bilinear')
        if has_mask:
            # draw a boundary(contour) in the original image separating mask and non-mask areas
            ax1.contour(y_train[ix].squeeze(), colors='k', linewidths=5, levels=[0.5])
        ax1.set_title('Input_image')

        ax2.imshow(y_train[ix].squeeze(), interpolation='bilinear')
        ax2.set_title('Mask')
        plt.show()



    # Re-size input data to 128x128 to match decoder-encoder depth
    def data_normalization(image, label):
        image = tf.image.resize_images(image, size=(128, 128))
        label = tf.image.resize_images(label, size=(128, 128))
        return image, label


    # Build a data iterator using DATASET API
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.map(data_normalization, num_parallel_calls=8)
    train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size)
    train_dataset = train_dataset.prefetch(buffer_size).repeat()

    validation_dataset = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
    validation_dataset = validation_dataset.map(data_normalization, num_parallel_calls=8)
    validation_dataset = validation_dataset.batch(1)
    validation_dataset = validation_dataset.prefetch(1)

    return train_dataset, validation_dataset
