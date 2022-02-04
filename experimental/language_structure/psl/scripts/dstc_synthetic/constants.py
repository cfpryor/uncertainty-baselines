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

RULE_WEIGHTS = [50.0, 50.0, 1.0]
RULE_NAMES = ('rule_1', 'rule_2', 'rule_3')

DATA_CONFIG = {
    'num_batches': 10,
    'batch_size': 1024,
    'max_dialog_size': 24,
    'max_utterance_size': 76,
    'num_labels': 39,
    'includes_word': -1,
    'excludes_word': -2,
    'utterance_mask': -1,
    'last_utterance_mask': -2,
    'pad_utterance_mask': -3,
    'mask_index': 0,
    'state_transitions': [[1, 3], [1, 7], [1, 1],
                          [2, 10], [2, 9], [2, 8],
                          [3, 5], [3, 12], [3, 15],
                          [4, 2], [4, 9], [4, 10],
                          [5, 11], [5, 14], [5, 17],
                          [6, 1], [6, 18], [6, 7],
                          [7, 5], [7, 12], [7, 15],
                          [8, 6], [8, 19], [8, 20],
                          [9, 1], [9, 3], [9, 7],
                          [10, 1], [10, 3], [10, 7],
                          [11, 8], [11, 13], [11, 2],
                          [12, 8], [12, 13], [12, 2],
                          [13, 6], [13, 19], [13, 20],
                          [14, 2], [14, 9], [14, 10],
                          [15, 2], [15, 9], [15, 10],
                          [16, 2], [16, 9], [16, 10],
                          [17, 4],
                          [18, 4],
                          [19, 6], [19, 20], [19, 19],
                          [20, 4],
                          [21, 5], [21, 12], [21, 17],
                          [22, 4],
                          [23, 5], [23, 12], [23, 17],
                          [24, 9], [24, 2], [24, 10],
                          [25, 4],
                          [26, 2], [26, 9], [26, 10],
                          [27, 4],
                          [28, 17], [28, 11], [28, 8],
                          [29, 14], [29, 35], [29, 37],
                          [30, 6], [30, 32], [30, 19],
                          [31, 17], [31, 14],
                          [32, 14],
                          [33, 6], [33, 19], [33, 20],
                          [34, 30], [34, 33], [34, 20],
                          [35, 6], [35, 20], [35, 29],
                          [36, 30], [36, 33],
                          [37, 6], [37, 19], [37, 29],
                          [38, 9], [38, 2], [38, 21]],
    'words': {
        '21': {
            'usr': {
                'index': 1,
                'words': ["$", "balance", "empty", "account", "savings", "slot", "checking"],
            },
            'sys': {
                'index': 2,
                'words': ["check", "checking", "balance", "savings", "account", "show", "available"],
            },
        },
        '23': {
            'usr': {
                'index': 3,
                'words': ["$", "takes", "found", "takes", "off"],
            },
            'sys': {
                'index': 4,
                'words': [],
            },
        },
    },
}

KWARGS_DICT = {
    'inference.constrained_gradient_decoding.ConstrainedGradientDecoding':
        {'alpha': 1000000, 'grad_steps': 100},
    'inference.unconstrained_inference.UnconstrainedInference':
        {},
    'learning.constrained_regularized_learning.ConstrainedRegularizedLearning':
        {'epochs': 100},
    'learning.unconstrained_learning.UnconstrainedLearning':
        {'epochs': 100},
    'models.dstc_synthetic.psl_model.PSLModelDSTCSynthetic':
        {'rule_weights': RULE_WEIGHTS, 'rule_names': RULE_NAMES, 'config': DATA_CONFIG},
    'scripts.dstc_synthetic.model_util':
        {'input_size': [DATA_CONFIG['max_dialog_size'], DATA_CONFIG['max_utterance_size']]},
}
