# coding=utf-8
# Copyright 2022 The Uncertainty Baselines Authors.
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

"""Corrupted Cifar100 Dataset."""

import os
from typing import Optional

from robustness_metrics.common import types
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from uncertainty_baselines.datasets import base

_DESCRIPTION = """\
Cifar100Corrupted is a dataset generated by adding 17 corruptions to the test
images in the Cifar100 dataset.
"""

_CITATION = """\
@inproceedings{
  hendrycks2018benchmarking,
  title={Benchmarking Neural Network Robustness to Common Corruptions and Perturbations},
  author={Dan Hendrycks and Thomas Dietterich},
  booktitle={International Conference on Learning Representations},
  year={2019},
  url={https://openreview.net/forum?id=HJz6tiCqYm},
}
"""

_CIFAR_IMAGE_SIZE = (32, 32, 3)
_CIFAR_CLASSES = 100
_CORRUPTIONS = [
    'brightness',
    'contrast',
    'defocus_blur',
    'elastic_transform',
    'fog',
    'frost',
    'glass_blur',  # called frosted_glass_blur in CIFAR-10
    'gaussian_blur',
    'gaussian_noise',
    'impulse_noise',
    'jpeg_compression',
    'pixelate',
    'saturate',  # not in standard set for CIFAR-10
    'shot_noise',
    'spatter',
    'speckle_noise',  # not in standard set for CIFAR-10
    'zoom_blur',
]
_NUM_EXAMPLES = 50000


class Cifar100CorruptedConfig(tfds.core.BuilderConfig):
  """BuilderConfig for Cifar100Corrupted."""

  def __init__(self, *, corruption_type, severity, **kwargs):
    """Constructor.

    Args:
      corruption_type: string, must be one of the items in _CORRUPTIONS.
      severity: integer, bewteen 1 and 5.
      **kwargs: keyword arguments forwarded to super.
    """
    super().__init__(**kwargs)
    self.corruption = corruption_type
    self.severity = severity


def _make_builder_configs():
  """Construct a list of BuilderConfigs.

  Construct a list of 85 Cifar100CorruptedConfig objects, corresponding to
  the 17 corruption types and 5 severities.

  Returns:
    A list of 85 Cifar100CorruptedConfig objects.
  """
  config_list = []
  for corruption in _CORRUPTIONS:
    for severity in range(1, 6):
      config_list.append(
          Cifar100CorruptedConfig(
              name=corruption + '_' + str(severity),
              description='Corruption method: ' + corruption +
              ', severity level: ' + str(severity),
              corruption_type=corruption,
              severity=severity,
          ))
  return config_list


class _Cifar100CorruptedDatasetBuilder(tfds.core.DatasetBuilder):
  """Corrupted Cifar100 dataset."""
  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }
  BUILDER_CONFIGS = _make_builder_configs()

  def __init__(self, data_dir, **kwargs):
    super().__init__(
        data_dir=data_dir, **kwargs)
    # We have to override self._data_dir to prevent the parent class from
    # appending the class name and version.
    self._data_dir = data_dir

  def _info(self):
    """Returns basic information of dataset.

    Returns:
      tfds.core.DatasetInfo.
    """
    info = tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Tensor(shape=_CIFAR_IMAGE_SIZE,
                                          dtype=tf.int64),
            'label': tfds.features.ClassLabel(num_classes=_CIFAR_CLASSES),
        }),
        supervised_keys=('image', 'label'),
        homepage='https://github.com/hendrycks/robustness',
        citation=_CITATION)

    split_infos = [
        tfds.core.SplitInfo(
            name=tfds.Split.TEST,
            shard_lengths=[_NUM_EXAMPLES],
            num_bytes=0,
        ),
    ]
    split_dict = tfds.core.SplitDict(split_infos, dataset_name=self.name)
    info.set_splits(split_dict)
    return info

  def _download_and_prepare(self, dl_manager, download_config=None):
    """Downloads and prepares dataset for reading."""
    raise NotImplementedError(
        'Must provide a data_dir with the files already downloaded to.')

  def _as_dataset(
      self,
      split: tfds.Split,
      decoders=None,
      read_config=None,
      shuffle_files=False) -> tf.data.Dataset:
    """Constructs a `tf.data.Dataset`."""
    del decoders
    del read_config
    del shuffle_files
    if split == tfds.Split.TEST:
      filename = (f'{self._builder_config.corruption}-'
                  f'{self._builder_config.severity}.tfrecords')
      filepath = os.path.join(self._data_dir, filename)
      dataset = tf.data.TFRecordDataset(filepath, buffer_size=16 * 1000 * 1000)
      return dataset
    raise ValueError('Unsupported split given: {}.'.format(split))


class Cifar100CorruptedDataset(base.BaseDataset):
  """CIFAR100-C dataset builder class."""

  def __init__(
      self,
      corruption_type: str,
      severity: int,
      split: str,
      num_parallel_parser_calls: int = 64,
      drop_remainder: bool = True,
      normalize: bool = True,
      download_data: bool = False,
      data_dir: Optional[str] = None,
  ):
    """Create a CIFAR100-C tf.data.Dataset builder.

    Args:
      corruption_type: Corruption name.
      severity: Corruption severity, an integer between 1 and 5.
      split: a dataset split, either a custom tfds.Split or one of the
        tfds.Split enums [TRAIN, VALIDAITON, TEST] or their lowercase string
        names.
      num_parallel_parser_calls: the number of parallel threads to use while
        preprocessing in tf.data.Dataset.map().
      drop_remainder: whether or not to drop the last batch of data if the
        number of points is not exactly equal to the batch size. This option
        needs to be True for running on TPUs.
      normalize: whether or not to normalize each image by the CIFAR dataset
        mean and stddev.
      download_data: Whether or not to download data before loading. Currently
        unsupported.
      data_dir: Path to a directory containing the CIFAR dataset, with
        filenames '{corruption_name}_{corruption_severity}.tfrecords'.
    """
    self._normalize = normalize
    dataset_builder = _Cifar100CorruptedDatasetBuilder(
        data_dir, config=f'{corruption_type}_{severity}')
    super().__init__(
        name=f'{dataset_builder.name}/{dataset_builder.builder_config.name}',
        dataset_builder=dataset_builder,
        split=split,
        num_parallel_parser_calls=num_parallel_parser_calls,
        drop_remainder=drop_remainder,
        download_data=download_data)

  def _create_process_example_fn(self) -> base.PreProcessFn:

    def _example_parser(example: types.Features) -> types.Features:
      """A pre-process function to return images in [0, 1]."""
      features = tf.io.parse_single_example(
          example['features'],
          features={
              'image': tf.io.FixedLenFeature([], tf.string),
              'label': tf.io.FixedLenFeature([], tf.int64),
          })
      dtype = tf.float32
      image = tf.io.decode_raw(features['image'], tf.uint8)
      image = tf.cast(tf.reshape(image, [32, 32, 3]), dtype)
      image = tf.image.convert_image_dtype(image, dtype)
      image = image / 255  # to convert into the [0, 1) range
      if self._normalize:
        # We use the convention of mean = np.mean(train_images, axis=(0,1,2))
        # and std = np.std(train_images, axis=(0,1,2)).
        mean = tf.constant([0.4914, 0.4822, 0.4465], dtype=dtype)
        std = tf.constant([0.2470, 0.2435, 0.2616], dtype=dtype)
        # Previously, std = np.mean(np.std(train_images, axis=(1, 2)), axis=0)
        # which gave std = tf.constant([0.2023, 0.1994, 0.2010], dtype=dtype).
        # However, we change convention to use the std over the entire training
        # set instead.
        image = (image - mean) / std

      label = tf.cast(features['label'], dtype)
      return {
          'features': image,
          'labels': tf.cast(label, tf.int32),
      }

    return _example_parser
