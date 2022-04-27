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
"""Differentiable PSL constraints.

File consists of:
- Differentiable PSL constraints for dialog structure rules.
"""

from typing import List

import tensorflow as tf
from models import abstract_psl_model


class PSLModelDSTCAlteredSynthetic(abstract_psl_model.PSLModel):
    """Defining PSL rules for the DSTC Altered Synthetic dataset."""

    def __init__(self, rule_weights: List[float], rule_names: List[str],
                 **kwargs) -> None:
        super().__init__(rule_weights, rule_names, **kwargs)

        self.config = kwargs['config']
        self.batch_size = self.config['batch_size']
        self.dialog_size = self.config['max_dialog_size']
        self.predicates = {}

    def rule_1(self, logits, **unused_kwargs) -> float:
        """weight: has_word -> state"""

        logits = tf.reshape(logits, (-1, self.dialog_size, logits.shape[-1]))
        words = self.predicates['words']
        weights = self.config['pos_psl_weights']

        repeat_words = tf.repeat(words, logits.shape[-1], axis=2)
        reshape_words = tf.reshape(repeat_words, (-1, self.dialog_size, words.shape[-1], logits.shape[-1]))

        repeat_logits = tf.repeat(logits, words.shape[-1], axis=1)
        reshape_logits = tf.reshape(repeat_logits, (-1, self.dialog_size, words.shape[-1], logits.shape[-1]))

        return_loss = self.template_rx_implies_sx(reshape_words, reshape_logits)

        return weights * return_loss

    def rule_2(self, logits, **unused_kwargs) -> float:
        """-weight: has_word -> !state"""

        logits = tf.reshape(logits, (-1, self.dialog_size, logits.shape[-1]))
        words = self.predicates['words']
        weights = self.config['neg_psl_weights']

        repeat_words = tf.repeat(words, logits.shape[-1], axis=2)
        reshape_words = tf.reshape(repeat_words, (-1, self.dialog_size, words.shape[-1], logits.shape[-1]))

        repeat_logits = tf.repeat(logits, words.shape[-1], axis=1)
        reshape_logits = tf.reshape(repeat_logits, (-1, self.dialog_size, words.shape[-1], logits.shape[-1]))

        return_loss = self.template_rx_implies_sx(reshape_words, self.soft_not(reshape_logits))

        return weights * return_loss

    def generate_predicates(self, data: tf.Tensor):
        self.predicates['words'] = tf.cast(data, tf.float32)

    def compute_loss_per_rule(self, data: tf.Tensor,
                              logits: tf.Tensor) -> List[float]:
        """Calculate the loss for each of the PSL rules."""
        rule_kwargs = dict(logits=logits, data=data)
        losses = []

        for rule_weight, rule_function in zip(self.rule_weights,
                                              self.rule_functions):
            losses.append(rule_weight * rule_function(**rule_kwargs))
        return losses

    def compute_loss(self, data: tf.Tensor, logits: tf.Tensor) -> float:
        """Calculate the total loss for all PSL rules."""
        return tf.reduce_sum(self.compute_loss_per_rule(data, logits))

    def set_batch_size(self, batch_size: float) -> None:
        self.batch_size = batch_size

    def set_dialog_size(self, dialog_size: float) -> None:
        self.dialog_size = dialog_size
