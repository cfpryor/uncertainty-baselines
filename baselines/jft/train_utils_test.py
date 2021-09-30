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

"""Tests for training utilities used in the ViT experiments."""

import jax
import tensorflow as tf
import train_utils  # local file import


class TrainUtilsTest(tf.test.TestCase):

  def test_sigmoid_xent(self):
    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)
    n = 5
    logits = jax.random.normal(key1, shape=(n,))
    labels = jax.random.bernoulli(key2, shape=(n,))
    expected_loss = 5.22126
    actual_loss = train_utils.sigmoid_xent(logits=logits, labels=labels)
    self.assertAllClose(actual_loss, expected_loss)

  def test_softmax_xent(self):
    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)
    n = 5
    k = 3
    logits = jax.random.normal(key1, shape=(n, k))
    labels = jax.nn.one_hot(
        jax.random.randint(key2, shape=(n,), minval=0, maxval=k - 1),
        num_classes=k)
    expected_loss = 2.55749
    actual_loss = train_utils.sigmoid_xent(logits=logits, labels=labels)
    self.assertAllClose(actual_loss, expected_loss)

  def test_accumulate_gradient(self):
    # TODO(dusenberrymw): Add a test for this.
    pass

  def test_create_learning_rate_schedule(self):
    total_steps = 10
    base = 0.1
    decay_type = "linear"
    warmup_steps = 2
    linear_end = 1e-4
    lr_fn = train_utils.create_learning_rate_schedule(
        total_steps,
        base=base,
        decay_type=decay_type,
        warmup_steps=warmup_steps,
        linear_end=linear_end)
    expected_lrs = [
        0.0, 0.05000000074505806, 0.10000000149011612, 0.08751250058412552,
        0.07502499967813492
    ]
    actual_lrs = [float(lr_fn(i)) for i in range(5)]
    self.assertAllClose(actual_lrs, expected_lrs)

    decay_type = "cosine"
    expected_lrs = [0., 0.05, 0.1, 0.087513, 0.075025]
    actual_lrs = [float(lr_fn(i)) for i in range(5)]
    self.assertAllClose(actual_lrs, expected_lrs)

  def test_tree_map_with_regex(self):
    d = {"this": 1, "that": {"another": 2, "wow": 3, "cool": {"neat": 4}}}

    f = lambda x, _: x + 1
    regex_rules = [(".*anot.*", 1)]
    mapped_d_expected = {
        "this": 1,
        "that": {
            "another": 3,
            "wow": 3,
            "cool": {
                "neat": 4
            }
        }
    }
    mapped_d_actual = train_utils.tree_map_with_regex(f, d, regex_rules)
    self.assertEqual(mapped_d_actual, mapped_d_expected)

    regex_rules = [(".*that.*", 1)]
    mapped_d_expected = {
        "this": 1,
        "that": {
            "another": 3,
            "wow": 4,
            "cool": {
                "neat": 5
            }
        }
    }
    mapped_d_actual = train_utils.tree_map_with_regex(f, d, regex_rules)
    self.assertEqual(mapped_d_actual, mapped_d_expected)

  def test_itstime(self):
    self.assertTrue(
        train_utils.itstime(
            step=1, every_n_steps=2, total_steps=4, last=True, first=True))
    self.assertTrue(
        train_utils.itstime(
            step=1, every_n_steps=2, total_steps=4, last=False, first=True))
    self.assertTrue(
        train_utils.itstime(
            step=2, every_n_steps=2, total_steps=4, last=True, first=True))
    self.assertTrue(
        train_utils.itstime(
            step=4, every_n_steps=2, total_steps=4, last=True, first=True))
    self.assertTrue(
        train_utils.itstime(
            step=4, every_n_steps=2, total_steps=4, last=True, first=False))
    self.assertFalse(
        train_utils.itstime(
            step=1, every_n_steps=2, total_steps=4, last=True, first=False))
    self.assertFalse(
        train_utils.itstime(
            step=4, every_n_steps=3, total_steps=4, last=False, first=True))


if __name__ == "__main__":
  tf.test.main()
