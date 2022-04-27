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
import models.dstc_altered_synthetic.psl_model_test_util as test_util

from models.dstc_altered_synthetic.psl_model import PSLModelDSTCAlteredSynthetic


class PslRulesTest(tf.test.TestCase):

    def setUp(self):
        super(PslRulesTest, self).setUp()

        self.config = test_util.DATA_CONFIG
        self.features = test_util.FEATURES
        self.logits = tf.constant(test_util.LOGITS)

        self.constraints = PSLModelDSTCAlteredSynthetic([], [], config=self.config)

    def test_psl_rule_1(self):
        # Total = (10 * 1) + (20 * 1) + (30 * 0.5) + (40 * 0.5) + (60 * 1) +
        #         (70 * 1) + (80 * 1) + (10 * 1) + (20 * 0.7) + (30 * 0.3) +
        #         (40 * 1) = 348
        self.constraints.generate_predicates(self.features)
        loss = tf.reduce_sum(self.constraints.rule_1(logits=self.logits))
        self.assertNear(loss, 348.0, err=1e-6)

    if __name__ == '__main__':
        tf.test.main()