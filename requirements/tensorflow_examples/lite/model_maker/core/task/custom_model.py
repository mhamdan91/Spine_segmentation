# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Base custom model that is already retained by data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import os
import tempfile

import tensorflow.compat.v2 as tf
from tensorflow_examples.lite.model_maker.core import compat
from tensorflow_examples.lite.model_maker.core import model_export_format as mef

DEFAULT_QUANTIZATION_STEPS = 2000


def get_representative_dataset_gen(dataset, num_steps):

  def representative_dataset_gen():
    """Generates representative dataset for quantized."""
    for image, _ in dataset.take(num_steps):
      yield [image]

  return representative_dataset_gen


class CustomModel(abc.ABC):
  """"The abstract base class that represents a Tensorflow classification model."""

  def __init__(self, model_export_format, model_spec, shuffle):
    """Initialize a instance with data, deploy mode and other related parameters.

    Args:
      model_export_format: Model export format such as saved_model / tflite.
      model_spec: Specification for the model.
      shuffle: Whether the data should be shuffled.
    """
    if model_export_format != mef.ModelExportFormat.TFLITE:
      raise ValueError('Model export format %s is not supported currently.' %
                       str(model_export_format))

    self.model_export_format = model_export_format
    self.model_spec = model_spec
    self.shuffle = shuffle
    self.model = None

  def preprocess(self, sample_data, label):
    """Preprocess the data."""
    # TODO(yuqili): remove this method once preprocess for image classifier is
    # also moved to DataLoader part.
    return sample_data, label

  @abc.abstractmethod
  def train(self, train_data, validation_data=None, **kwargs):
    return

  @abc.abstractmethod
  def export(self, **kwargs):
    return

  def summary(self):
    self.model.summary()

  @abc.abstractmethod
  def evaluate(self, data, **kwargs):
    return

  def _gen_dataset(self,
                   data,
                   batch_size=32,
                   is_training=True,
                   input_pipeline_context=None):
    """Generates training / validation dataset."""
    # The dataset is always sharded by number of hosts.
    # num_input_pipelines is the number of hosts rather than number of cores.
    ds = data.dataset
    if input_pipeline_context and input_pipeline_context.num_input_pipelines > 1:
      ds = ds.shard(input_pipeline_context.num_input_pipelines,
                    input_pipeline_context.input_pipeline_id)

    ds = ds.map(
        self.preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if is_training:
      if self.shuffle:
        ds = ds.shuffle(buffer_size=min(data.size, 100))
      ds = ds.repeat()

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds

  def _export_tflite(self,
                     tflite_filename,
                     quantized=False,
                     quantization_steps=None,
                     representative_data=None):
    """Converts the retrained model to tflite format and saves it.

    Args:
      tflite_filename: File name to save tflite model.
      quantized: boolean, if True, save quantized model.
      quantization_steps: Number of post-training quantization calibration steps
        to run. Used only if `quantized` is True.
      representative_data: Representative data used for post-training
        quantization. Used only if `quantized` is True.
    """
    temp_dir = None
    if compat.get_tf_behavior() == 1:
      temp_dir = tempfile.TemporaryDirectory()
      save_path = os.path.join(temp_dir.name, 'saved_model')
      self.model.save(save_path, include_optimizer=False, save_format='tf')
      converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model(save_path)
    else:
      converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

    if quantized:
      if quantization_steps is None:
        quantization_steps = DEFAULT_QUANTIZATION_STEPS
      if representative_data is None:
        raise ValueError(
            'representative_data couldn\'t be None if model is quantized.')
      ds = self._gen_dataset(
          representative_data, batch_size=1, is_training=False)
      converter.representative_dataset = tf.lite.RepresentativeDataset(
          get_representative_dataset_gen(ds, quantization_steps))

      converter.optimizations = [tf.lite.Optimize.DEFAULT]
      converter.inference_input_type = tf.uint8
      converter.inference_output_type = tf.uint8
      converter.target_spec.supported_ops = [
          tf.lite.OpsSet.TFLITE_BUILTINS_INT8
      ]
    tflite_model = converter.convert()
    if temp_dir:
      temp_dir.cleanup()

    with tf.io.gfile.GFile(tflite_filename, 'wb') as f:
      f.write(tflite_model)

    tf.compat.v1.logging.info('Export to tflite model in %s.', tflite_filename)
