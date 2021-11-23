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
-
"""
from typing import List

import math

import tensorflow as tf
from inference.abstract_inference_application import AbstractInferenceApplication


class ConstrainedBeamSearchDecoding(AbstractInferenceApplication):
    """Constrained beam search decoding."""

    def __init__(self, model, constraints, **kwargs) -> None:
        super().__init__(model, constraints, **kwargs)

        if 'num_beams' not in kwargs:
            raise KeyError('Missing argument: alpha')
        self.num_beams = kwargs['num_beams']

        self.beams = []

    def predict(self, dataset: tf.Tensor) -> List[tf.Tensor]:
        """Constrained prediction using beam search decoding."""
        self.beams.clear()
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
        logits = self.model(data, training=False)
        batch_beams = []

        for example in logits:
            batch_beams.append([[[], 0.0]])
            for distribution in example:
                candidates = []
                for index_i in range(len(batch_beams[-1])):
                    sequence, value = batch_beams[-1][index_i]
                    for index_j in range(len(distribution)):
                        candidates.append([sequence + [index_j], value - math.log(distribution[index_j])])
                ordered_candidates = sorted(candidates, key=lambda seq: seq[1])
                batch_beams[-1] = ordered_candidates[:self.num_beams]
        return batch_beams
