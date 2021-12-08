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
from learning.abstract_learning_application import AbstractLearningApplication


class ConstrainedRegularizedLearning(AbstractLearningApplication):
    """Constrained regularized learning."""

    def __init__(self, model, constraints, epochs, **kwargs) -> None:
        super().__init__(model, constraints, epochs, **kwargs)

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
        self.constraints.generate_predicates(data)

        with tf.GradientTape() as tape:
            logits = self.model(data, training=True)
            constraint_loss = self.constraints.compute_total_loss(data, logits)
            loss = self.model.compiled_loss(labels, logits, regularization_losses=self.model.losses)
            totalLoss = loss + constraint_loss

        gradients = tape.gradient(totalLoss, self.model.trainable_variables)

        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.model.compiled_metrics.update_state(labels, logits)

        return {m.name: m.result() for m in self.model.metrics}
