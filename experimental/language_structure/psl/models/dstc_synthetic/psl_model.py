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


class PSLModelDSTCSynthetic(abstract_psl_model.PSLModel):
    """Defining PSL rules for the DSTC Synthetic dataset."""

    def __init__(self, rule_weights: List[float], rule_names: List[str],
                 **kwargs) -> None:
        super().__init__(rule_weights, rule_names, **kwargs)

        if 'config' not in kwargs:
            raise KeyError('Missing argument: config')
        self.config = kwargs['config']
        self.batch_size = self.config['batch_size']
        self.dialog_size = self.config['max_dialog_size']

    def _get_tensor_column(self, data, index):
        """Gathers a column in a tensor and reshapes."""
        return tf.reshape(tf.gather(data, index, axis=-1), [self.batch_size, -1])

    def _first_statement(self):
        """Creates a (batch_size, dialog_size) first statement mask."""
        return tf.constant([[1.0] + [0.0] * (self.dialog_size - 1)] * self.batch_size)

    def rule_1(self, logits, **unused_kwargs) -> float:
        """Dialog structure rule.

        Rule:
          FirstStatement -> State("INFORM_INTENT -> REQUEST")

        Meaning:
          IF: the utterance is the first utterance in a dialog.
          THEN: the utterance is likely to belong to INFORM_INTENT -> REQUEST.

        Args:
          logits: logits outputted by a neural model.

        Returns:
          A loss incurred by this dialog structure rule.
        """
        first_statement = self.predicates['first_statement']
        state_greet = self._get_tensor_column(logits, 0)

        return self.template_rx_implies_sx(first_statement,
                                           state_greet)

    def generate_predicates(self, data: tf.Tensor):
        self.predicates['first_statement'] = self._first_statement()

    def compute_total_loss(self, data: tf.Tensor, logits: tf.Tensor) -> float:
        """Calculate the loss for all PSL rules."""
        total_loss = 0.0
        rule_kwargs = dict(logits=logits, data=data)

        for rule_weight, rule_function in zip(self.rule_weights, self.rule_functions):
            total_loss += rule_weight * tf.reduce_sum(rule_function(**rule_kwargs))

        return total_loss

    def compute_all_potential_losses(self, data: tf.Tensor, logits: tf.Tensor) -> List[float]:
        """Calculate the loss for each PSL rule."""
        rules_loss = []
        rule_kwargs = dict(logits=logits, data=data)

        for rule_weight, rule_function in zip(self.rule_weights, self.rule_functions):
            rules_loss.append(rule_weight * rule_function(**rule_kwargs))

        return rules_loss

    def set_batch_size(self, batch_size: float) -> None:
        self.batch_size = batch_size

    def set_dialog_size(self, dialog_size: float) -> None:
        self.dialog_size = dialog_size
