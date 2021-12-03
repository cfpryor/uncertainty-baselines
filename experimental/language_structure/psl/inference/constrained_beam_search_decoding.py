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
            raise KeyError('Missing argument: num_beams')
        if 'class_rules_mapping' not in kwargs:
            raise KeyError('Missing argument: class_rules_mapping')
        self.num_beams = kwargs['num_beams']
        self.class_rules_mapping = kwargs['class_rules_mapping']

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
        return self._beam_search(data, logits)

    def _beam_search(self, data: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
        batch_beams = []

        for index_i in range(len(logits)):
            batch_beams.append([[[], 0.0]])
            for distribution in logits[index_i]:
                candidates = []
                constrained_distribution = self._update_distribution(index_i, distribution, data, logits, len(logits[index_i]))
                for index_j in range(len(batch_beams[-1])):
                    sequence, value = batch_beams[-1][index_j]
                    for index_k in range(len(constrained_distribution)):
                        candidates.append([sequence + [index_k], value - math.log(constrained_distribution[index_k])])
                ordered_candidates = sorted(candidates, key=lambda seq: seq[1])
                batch_beams[-1] = ordered_candidates[:self.num_beams]
        return batch_beams

    def _update_distribution(self, index, distribution, data, logits, size):
        return distribution
        constraint_losses = self.constraints.compute_all_potential_losses(data, logits)
        masked_constraint_losses = []
        base_loss = 0.0

        for constraint_index in range(len(constraint_losses)):
            previous_mask = self._create_mask(index, size, len(constraint_losses[constraint_index].shape) - 1)
            current_mask = self._create_mask(index + 1, size, len(constraint_losses[constraint_index].shape) - 1) - previous_mask

            masked_constraint_losses.append(constraint_losses[constraint_index] * current_mask)
            rule_index, parity = self.class_rules_mapping[constraint_index]
            base_loss += parity * tf.reduce_sum(constraint_losses[constraint_index] * previous_mask)

        for class_index in self.class_rules_mapping:
            for rule_index, parity in self.class_rules_mapping[class_index]:
                distribution[class_index] += parity * masked_constraint_losses[rule_index]

        return distribution + base_loss

    def _create_mask(self, index, size, dimension):
        mask = tf.constant(([1.0] * index + [0.0] * (size - index)), dtype=tf.float32)
        if dimension == 2:
            mask = tf.cast(tf.sequence_mask(mask, size), tf.float32)
        return mask
