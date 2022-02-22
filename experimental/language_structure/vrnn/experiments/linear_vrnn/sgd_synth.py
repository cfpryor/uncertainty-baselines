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

r"""Vizier for linear VRNN for SimDialDataset.

"""

import default_config  # local file import from experimental.language_structure.vrnn.experiments.linear_vrnn

_DATASET = 'sgd_synth'


def get_config():
  """Returns the configuration for this experiment."""
  config = default_config.get_config(_DATASET)
  config.platform = 'df'
  config.tpu_topology = '4x4'

  config.train_epochs = 20

  return config


def get_sweep(hyper):
  """Returns hyperparameter sweep."""
  domain = [
      hyper.sweep('config.model.vae_cell.encoder_hidden_size',
                  hyper.discrete([200, 300])),
      hyper.sweep('config.base_learning_rate', hyper.discrete([5e-4, 1e-3]))
  ]
  sweep = hyper.product(domain)
  return sweep
