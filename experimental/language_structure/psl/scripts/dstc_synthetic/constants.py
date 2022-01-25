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

"""DSTC Synthetic constants."""

RULE_WEIGHTS = [1.0]
RULE_NAMES = ('rule_1')

DATA_CONFIG = {
    'num_batches': 5,
    'batch_size': 128,
    'max_dialog_size': 10,
    'max_utterance_size': 40,
    '3': {'sys': [],
          'usr': []},
    '7': {'sys': [],
          'usr': []},
    '21': {'sys': ["check", "checking", "balance", "savings", "account", "show", "available"],
           'usr': ["$", "balance", "empty", "account", "savings", "slot", "checking"]},
    '23': {'sys': [],
           'usr': ["$", "takes", "found", "takes", "off"]},
    'includes_word': -1,
    'excludes_word': -2,
    'utterance_mask': -1,
    'last_utterance_mask': -2,
    'pad_utterance_mask': -3,
}

KWARGS_DICT = {
    'inference.constrained_beam_search_decoding.ConstrainedBeamSearchDecoding':
        {'num_beams': 3, 'class_rules_parities': [-1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1],
         'class_rules_mapping': {0: [7], 1: [8], 2: [6], 3: [0, 1], 4: [9], 5: [2, 4, 5], 6: [10], 7: [3], 8: [11]}},
    'inference.constrained_gradient_decoding.ConstrainedGradientDecoding':
        {'alpha': 0.1, 'grad_steps': 25},
    'inference.unconstrained_inference.UnconstrainedInference':
        {},
    'learning.constrained_regularized_learning.ConstrainedRegularizedLearning':
        {'epochs': 100},
    'learning.unconstrained_learning.UnconstrainedLearning':
        {'epochs': 1},
    'models.multiwoz_synthetic.psl_model.PSLModelMultiWoZ':
        {'rule_weights': RULE_WEIGHTS, 'rule_names': RULE_NAMES, 'config': DATA_CONFIG},
    'scripts.multiwoz_synthetic.model_util':
        {'input_size': [DATA_CONFIG['max_dialog_size'], DATA_CONFIG['max_utterance_size']]},
}
