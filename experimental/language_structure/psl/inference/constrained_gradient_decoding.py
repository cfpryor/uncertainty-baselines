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
"""Constrained gradient decoding.

File consists of:
- Gradient updates during inference
"""
from typing import List

import tensorflow as tf
from inference.abstract_inference_application import AbstractInferenceApplication

class ConstrainedGradientDecoding(AbstractInferenceApplication):
    """Constrained gradient decoding."""

    def __init__(self, model, constraints, **kwargs) -> None:
        super().__init__(model, constraints, **kwargs)

        if 'alpha' not in kwargs:
            raise KeyError('Missing argument: alpha')
        if 'grad_steps' not in kwargs:
            raise KeyError('Missing argument: grad_steps')
        self.alpha = kwargs['alpha']
        self.grad_steps = kwargs['grad_steps']

        self.weights_copy = []
        self.logits = []

    def predict(self, dataset: tf.Tensor) -> List[tf.Tensor]:
        """Constrained prediction using gradient decoding."""
        self.logits.clear()
        self._copy_weights()

        for data_batch, label_batch in dataset:
            self.logits.append(self.batch_predict(data_batch, label_batch))
        self.weights_copy.clear()

        return self.logits

    def batch_predict(self, data: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
        """Batch constrained prediction using gradient decoding.

          Args:
            data: input features
            labels: ground truth labels

          Returns:
            Logits for a batch.
        """
        self._satisfy_constraints(data, labels)

        batch_logits = self.model(data, training=False)
        self.model.compiled_loss(labels, batch_logits)
        self.model.compiled_metrics.update_state(labels, batch_logits)

        self._reset_weights()

        return batch_logits

    def _satisfy_constraints(self, data: tf.Tensor, labels: tf.Tensor,):
        """Update weights to satisfy constraints while staying close to original weights."""
        for _ in range(self.grad_steps):
            with tf.GradientTape() as tape:
                logits = self.model(data, training=False)
                constraint_loss = self.constraints.compute_total_loss(data, logits)
                weight_loss = tf.reduce_sum([
                    tf.reduce_mean(tf.math.squared_difference(w, w_h))
                    for w, w_h in zip(self.weights_copy, self.model.weights)
                ])
                loss = constraint_loss + self.alpha * weight_loss

            gradients = tape.gradient(loss, self.model.trainable_variables)

            self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            self.model.compiled_metrics.update_state(labels, logits)

    def _copy_weights(self):
        """Copies model weights."""
        for layer in self.model.weights:
            self.weights_copy.append(tf.identity(layer))

    def _reset_weights(self):
        self.model.set_weights(self.weights_copy)
