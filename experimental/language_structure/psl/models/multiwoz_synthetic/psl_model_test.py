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
"""Tests for MultiWoz rules."""

import tensorflow as tf
import scripts.multiwoz_synthetic.data_util as data_util
import models.multiwoz_synthetic.psl_model_test_util as test_util

from models.multiwoz_synthetic.psl_model import PSLModelMultiWoZ
from inference.constrained_gradient_decoding import ConstrainedGradientDecoding


class PslRulesTest(tf.test.TestCase):

    def setUp(self):
        super(PslRulesTest, self).setUp()
        # Load data
        self.config = test_util.DATA_CONFIG
        self.data = test_util.DATA
        self.features = test_util.FEATURES
        self.logits = tf.constant(test_util.LOGITS)

        # Load parameters
        self.seed = test_util.SEED
        self.alpha = test_util.ALPHA
        self.grad_steps = test_util.GRAD_STEPS
        self.epochs = test_util.TRAINING_EPOCHS
        self.model_shape = [self.config['max_dialog_size'], self.config['max_utterance_size']]

        # Build dataset, constraints, and inference application
        self.train_ds, self.test_ds = data_util.prepare_dataset(self.data, self.config)
        self.constraints = PSLModelMultiWoZ([], [], config=self.config)
        self.inference = ConstrainedGradientDecoding(None, None, alpha=self.alpha, grad_steps=self.grad_steps)

        # Seed randomness
        tf.random.set_seed(self.seed)

    def _run_model(self, rule_names, weights):
        # Set rule functions and rule weights
        self.constraints.set_batch_size(4)
        for test_features, _ in self.test_ds:
            self.constraints.generate_predicates(test_features)
        self.constraints.set_rule_functions(self.constraints, rule_names)
        self.constraints.set_rule_weights(self.constraints, weights)

        # Set constraints and a fresh neural model
        model = test_util.build_model(self.model_shape)
        model.fit(self.train_ds, epochs=self.epochs)
        self.inference.set_constraints(self.inference, self.constraints)
        self.inference.set_model(self.inference, model)

        # Run constrained inference
        logits = self.inference.predict(self.test_ds)
        return tf.math.argmax(logits[0], axis=-1)

    def test_psl_rule_1_run_model(self):
        predictions = self._run_model(['rule_1'], [1.0])
        self.assertNotEqual(predictions[1][1], self.config['class_map']['greet'])
        self.assertNotEqual(predictions[1][2], self.config['class_map']['greet'])
        self.assertNotEqual(predictions[2][1], self.config['class_map']['greet'])
        self.assertNotEqual(predictions[2][2], self.config['class_map']['greet'])
        self.assertNotEqual(predictions[2][3], self.config['class_map']['greet'])

    def test_psl_rule_1(self):
        self.constraints.set_batch_size(2)
        self.constraints.generate_predicates(self.features)
        loss = tf.reduce_sum(self.constraints.rule_1(logits=self.logits))
        self.assertNear(loss, 1.4, err=1e-6)

    def test_psl_rule_2_run_model(self):
        predictions = self._run_model(['rule_2'], [1.0])
        self.assertEqual(predictions[2][0], self.config['class_map']['greet'])
        self.assertEqual(predictions[3][0], self.config['class_map']['greet'])

    def test_psl_rule_2(self):
        self.constraints.set_batch_size(2)
        self.constraints.generate_predicates(self.features)
        loss = tf.reduce_sum(self.constraints.rule_2(logits=self.logits))
        self.assertNear(loss, 0.6, err=1e-6)

    def test_psl_rule_3_run_model(self):
        predictions = self._run_model(['rule_3'], [1.0])
        self.assertEqual(predictions[0][0], self.config['class_map']['init_request'])
        self.assertEqual(predictions[1][0], self.config['class_map']['init_request'])

    def test_psl_rule_3(self):
        self.constraints.set_batch_size(2)
        self.constraints.generate_predicates(self.features)
        loss = tf.reduce_sum(self.constraints.rule_3(logits=self.logits))
        self.assertNear(loss, 0.8, err=1e-6)

    def test_psl_rule_4_run_model(self):
        predictions = self._run_model(['rule_4'], [1.0])
        self.assertEqual(predictions[1][1], self.config['class_map']['second_request'])
        self.assertEqual(predictions[2][1], self.config['class_map']['second_request'])

    def test_psl_rule_4(self):
        self.constraints.set_batch_size(2)
        self.constraints.generate_predicates(self.features)
        loss = tf.reduce_sum(self.constraints.rule_4(logits=self.logits))
        self.assertNear(loss, 1.8, err=1e-6)

    def test_psl_rule_5_run_model(self):
        predictions = self._run_model(['rule_5'], [1.0])
        self.assertNotEqual(predictions[1][1], self.config['class_map']['init_request'])
        self.assertNotEqual(predictions[2][1], self.config['class_map']['init_request'])

    def test_psl_rule_5(self):
        self.constraints.set_batch_size(2)
        self.constraints.generate_predicates(self.features)
        loss = tf.reduce_sum(self.constraints.rule_5(logits=self.logits))
        self.assertNear(loss, 1.4, err=1e-6)

    def test_psl_rule_6_run_model(self):
        predictions = self._run_model(['rule_6'], [1.0])
        self.assertNotEqual(predictions[1][0], self.config['class_map']['greet'])
        self.assertNotEqual(predictions[2][0], self.config['class_map']['greet'])

    def test_psl_rule_6(self):
        self.constraints.set_batch_size(2)
        self.constraints.generate_predicates(self.features)
        loss = tf.reduce_sum(self.constraints.rule_6(logits=self.logits))
        self.assertNear(loss, 1.4, err=1e-6)

    def test_psl_rule_7_run_model(self):
        predictions = self._run_model(['rule_7'], [1.0])
        self.assertEqual(predictions[1][2], self.config['class_map']['end'])
        self.assertEqual(predictions[2][3], self.config['class_map']['end'])

    def test_psl_rule_7(self):
        self.constraints.set_batch_size(2)
        self.constraints.generate_predicates(self.features)
        loss = tf.reduce_sum(self.constraints.rule_7(logits=self.logits))
        self.assertNear(loss, 1.1, err=1e-6)

    def test_psl_rule_8(self):
        self.constraints.set_batch_size(2)
        self.constraints.generate_predicates(self.features)
        loss = tf.reduce_sum(self.constraints.rule_8(logits=self.logits))
        self.assertNear(loss, 0.9, err=1e-6)

    def test_psl_rule_9(self):
        self.constraints.set_batch_size(2)
        self.constraints.generate_predicates(self.features)
        loss = tf.reduce_sum(self.constraints.rule_9(logits=self.logits))
        self.assertNear(loss, 0.8, err=1e-6)

    def test_psl_rule_10(self):
        self.constraints.set_batch_size(2)
        self.constraints.generate_predicates(self.features)
        loss = tf.reduce_sum(self.constraints.rule_10(logits=self.logits))
        self.assertNear(loss, 0.3, err=1e-6)

    def test_psl_rule_11(self):
        self.constraints.set_batch_size(2)
        self.constraints.generate_predicates(self.features)
        loss = tf.reduce_sum(self.constraints.rule_11(logits=self.logits))
        self.assertNear(loss, 0.7, err=1e-6)

    def test_psl_rule_12(self):
        self.constraints.set_batch_size(2)
        self.constraints.generate_predicates(self.features)
        loss = tf.reduce_sum(self.constraints.rule_12(logits=self.logits))
        self.assertNear(loss, 0.1, err=1e-6)


if __name__ == '__main__':
    tf.test.main()
