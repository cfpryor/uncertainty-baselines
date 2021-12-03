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
        if 'class_rules_parities' not in kwargs:
            raise KeyError('Missing argument: class_rules_parities')
        self.num_beams = kwargs['num_beams']
        self.class_rules_mapping = kwargs['class_rules_mapping']
        self.class_rules_parities = kwargs['class_rules_parities']

        self.beams = []

    def predict(self, dataset: tf.Tensor) -> List[tf.Tensor]:
        """Constrained prediction using beam search decoding."""
        self.beams.clear()
        self.constraints.set_batch_size(1)
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
        return self._batch_beam_search(data, logits)

    def _batch_beam_search(self, data: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
        batch_beams = []

        for dialog_data, dialog_logits in zip(data, logits):
            batch_beams.append(self._beam_search(dialog_data, dialog_logits))

        return batch_beams

    def _beam_search(self, dialog_data: tf.Tensor, dialog_logits: tf.Tensor):
        beams = [[[], 0.0]]

        for utterance_index in range(len(dialog_logits)):
            candidates = []
            utterance_logits = dialog_logits[utterance_index] + self._update_distribution(dialog_data, dialog_logits, utterance_index)
            for index_j in range(len(beams)):
                sequence, value = beams[index_j]
                for index_k in range(len(utterance_logits)):
                    if utterance_logits[index_k] < 0:
                        candidates.append([sequence + [index_k], value])
                    else:
                        candidates.append([sequence + [index_k], value - math.log(utterance_logits[index_k])])
            ordered_candidates = sorted(candidates, key=lambda seq: seq[1])
            beams = ordered_candidates[:self.num_beams]

        return beams

    def _update_distribution(self, data, logits, utterance_index):
        data_tmp = data[:utterance_index + 1]
        logits_tmp = logits[:utterance_index + 1]

        self.constraints.set_dialog_size(utterance_index + 1)
        self.constraints.generate_predicates(data_tmp)

        potential_losses = self.constraints.compute_all_potential_losses(data_tmp, logits_tmp)

        distribution_loss = [0.0] * len(logits[0])
        for rule_index in range(len(potential_losses)):
            parity = self.class_rules_parities[rule_index]

            if len(potential_losses[rule_index].shape) == 2:
                for class_index in self.class_rules_mapping:
                    distribution_loss[class_index] += parity * tf.reduce_sum(potential_losses[rule_index][:, :-1])
                    if rule_index in self.class_rules_mapping[class_index]:
                        distribution_loss[class_index] += parity * tf.reduce_sum(potential_losses[rule_index][:, -1:])
            else:
                for class_index in self.class_rules_mapping:
                    distribution_loss[class_index] += parity * tf.reduce_sum(potential_losses[rule_index][:, :-1, :])
                    if rule_index in self.class_rules_mapping[class_index]:
                        distribution_loss[class_index] += parity * tf.reduce_sum(potential_losses[rule_index][:, -1:, :])

        return distribution_loss
