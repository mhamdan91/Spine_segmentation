import  tensorflow as tf
from tensorflow.tensorflow_examples.models.pix2pix import pix2pix
'''
Build a U-NET image segmentation model.
A U-Net consists of an encoder (downsampler) and decoder (upsampler). The pre-trained 
MobileNetV2 model is used as an encoder, and the decoder will be an upsample 
block that is implemented in pix2pix from tensorflow
'''


# Base model - Encoder
base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
layers = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

down_stack.trainable = False

# Decoder network
up_stack = [
    pix2pix.upsample(128, 3),  # 4x4    -> 8x8
    pix2pix.upsample(64, 3),   # 8x8    -> 16x16
    pix2pix.upsample(32, 3),   # 16x16  -> 32x32
    pix2pix.upsample(8, 3),    # 32x32  -> 64x64
]


# Define the u-net model
def unet(output_channels):
  # Input shape, takes an RGB image of size 128x128
  inputs = tf.keras.layers.Input(shape=[128, 128, 3])
  x = inputs

  # Downsampling through the model
  skips = down_stack(x)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])

  # This is the last layer of the model
  last = tf.keras.layers.Conv2DTranspose(
      output_channels, 3, strides=2,
      padding='same')  #64x64 -> 128x128

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)
