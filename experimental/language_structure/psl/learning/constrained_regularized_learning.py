# coding=utf-8
# Copyright 2021 The Uncertainty Baselines Authors.
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

# Lint as: python3
"""Constrained regularized learning.

File consists of:
-
"""

import tensorflow as tf
from learning import abstract_learning_application


class ConstrainedRegularizedLearning(abstract_learning_application.AbstractLearningApplication):
    """Constrained regularized learning."""

    def __init__(self, model, constraints, **kwargs) -> None:
        super().__init__(model, constraints, **kwargs)

        if 'epochs' not in kwargs:
            raise KeyError('Missing argument: epochs')
        self.epochs = kwargs['epochs']

    def fit(self, dataset: tf.Tensor) -> None:
        """Constrained learning using gradients."""
        for _ in range(self.epochs):
            for data_batch, label_batch in dataset:
                self.batch_fit(data_batch, label_batch)

    def batch_fit(self, data: tf.Tensor, labels: tf.Tensor) -> None:
        """Batch constrained learning using gradients.

          Args:
            data: input features
            labels: ground truth labels
        """
        pass
