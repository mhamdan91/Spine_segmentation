from __future__ import absolute_import, division, print_function
import tensorflow as tf
import os
from termcolor import colored
from keras.callbacks import EarlyStopping, ModelCheckpoint

sep = os.sep
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
layers = tf.keras.layers
tf.enable_eager_execution(config=config)
tf.executing_eagerly()
print(tf.__version__)

###############################################
'''
This file represents the main training loop used to train segmentation model
1- Load data
2- Compile model
3- Fit model if training mode
'''

from model import unet_model as UNET # Import the u-net network
from utils import helper_fn as fn # import helper function to visualize data
from utils import data_loader # Import data loader


def train(batch_size=2, train_mode=2, epochs=2,visualize=0, checkpoint_path=None, images_path=None,masks_path=None ):
    """

    :param batch_size: consumed batch size by the network at each training step
    :param train_mode: determines training mode. If set to 0 then model is in prediction mode, if set to 1 then model will train from an existing
    checkpoint, and if set to 2 the model will train from scratch.
    :param epochs: number of forward and backward passes to train the model for.
    :param visualize: determines which dataset to visualize. 0 for training dataset and 1 for validation.
    :param checkpoint_path: path to existing checkpoint in case train_mode equals 0 or 1.
    :param images_path: path to images to be used for training or prediction
    :param masks_path:  path to ground truth masks to be used for training or prediction
    :return: None

    """



    train_dataset, validation_dataset = data_loader.data_loader(batch_size=batch_size, buffer_size=2, visualize=True,
                                                                images_path=images_path,masks_path=masks_path ) # load train and validation datasets

    OUTPUT_CHANNELS = 3 # Three channels to corrospond to the three possible labels for each pixel
    model = UNET.unet(OUTPUT_CHANNELS)
    model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    model.summary()

    callbacks = [
        EarlyStopping(patience=10, verbose=1),
        ModelCheckpoint('checkpoints'+sep+'pre_trained_1.h5', verbose=1, save_best_only=True, save_weights_only=True)
                ]

    # Predict using a trained model
    if train_mode == 0:
        if os.path.exists(checkpoint_path):
            model.load_weights(checkpoint_path)
            print(colored("Model Weights Loaded Successfully", "blue"))
        if visualize == 0:
            fn.show_predictions(train_dataset, model, 2)
        else:
            fn.show_predictions(validation_dataset, model, 2)


    else:
        # Check if train from a previous checkpoint or train from scratch
        if train_mode == 1:
            if os.path.exists(checkpoint_path):
                model.load_weights(checkpoint_path)
                print(colored("Model Weights Loaded Successfully", "blue"))

        model_history = model.fit(train_dataset, epochs=epochs,
                                  steps_per_epoch=10,
                                  validation_steps=1,
                                  validation_data=validation_dataset,
                                  callbacks=callbacks)
        fn.visualize_training(model_history)
        fn.show_predictions(train_dataset, model, 1)



