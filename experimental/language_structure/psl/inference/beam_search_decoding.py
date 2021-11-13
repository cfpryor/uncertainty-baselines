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
"""Constrained beam search decoding.

File consists of:
- Gradient updates during inference
"""

from typing import List

import tensorflow as tf
from inference import abstract_inference_application


class GradientDecoding(abstract_inference_application.AbstractInferenceApplication):
    """Constrained beam search decoding."""

    def __init__(self, model, constraints, **kwargs) -> None:
        super().__init__(model, constraints, **kwargs)
        self.beams = []

    def predict(self, dataset: tf.Tensor) -> List[tf.Tensor]:
        """Constrained prediction using beam search decoding."""
        self.clear_beams()
        for data_batch, label_batch in dataset:
            self.beams.append(self.batch_predict(data_batch, label_batch))

        return self.beams

    def batch_predict(self, data: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
        """Batch constrained prediction using beam search decoding.

          Args:
            data: input features
            labels: ground truth labels

          Returns:
            Logits for a batch.
        """
        batch_beams = None
        return batch_beams

    def clear_beams(self):
        self.beams = []