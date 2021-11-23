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
"""Unconstrained learning."""
from typing import List

import tensorflow as tf
from learning.abstract_learning_application import AbstractLearningApplication


class UnconstrainedLearning(AbstractLearningApplication):
    """Unconstrained learning."""

    def __init__(self, model, constraints, epochs, **kwargs) -> None:
        super().__init__(model, constraints, epochs, **kwargs)

    def fit(self, dataset) -> List[tf.Tensor]:
        """Unconstrained fit."""
        self.model.fit(dataset, epochs=self.epochs)

    def batch_fit(self, data: tf.Tensor) -> tf.Tensor:
        """Unconstrained batch fit."""
        pass
