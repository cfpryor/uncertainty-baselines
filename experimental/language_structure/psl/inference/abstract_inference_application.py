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

"""Abstract class for inference applications."""

import abc
from typing import List

import tensorflow as tf


class AbstractInferenceApplication(abc.ABC):
    """Abstract class for inference applications."""

    def __init__(self, model, constraints, **kwargs) -> None:
        self.model = model
        self.constraints = constraints
        self.kwargs = kwargs

    @abc.abstractmethod
    def predict(self, dataset) -> List[tf.Tensor]:
        pass

    @abc.abstractmethod
    def batch_predict(self, data: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
        pass
