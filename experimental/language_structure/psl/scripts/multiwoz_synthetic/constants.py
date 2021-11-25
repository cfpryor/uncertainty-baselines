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

"""MultiWoZ Synthetic constants."""

RULE_WEIGHTS = [1.0, 20.0, 5.0, 5.0, 5.0, 10.0, 5.0, 20.0, 5.0, 5.0, 5.0, 10.0]
RULE_NAMES = ('rule_1', 'rule_2', 'rule_3', 'rule_4', 'rule_5', 'rule_6',
              'rule_7', 'rule_8', 'rule_9', 'rule_10', 'rule_11', 'rule_12')

DATA_CONFIG = {
    'batch_size': 128,
    'max_dialog_size': 10,
    'max_utterance_size': 40,
    'class_map': {
        'accept': 0,
        'cancel': 1,
        'end': 2,
        'greet': 3,
        'info_question': 4,
        'init_request': 5,
        'insist': 6,
        'second_request': 7,
        'slot_question': 8,
    },
    'accept_words': ['yes', 'great'],
    'cancel_words': ['no'],
    'end_words': ['thank', 'thanks'],
    'greet_words': ['hello', 'hi'],
    'info_question_words': ['address', 'phone'],
    'insist_words': ['sure', 'no'],
    'slot_question_words': ['what', '?'],
    'includes_word': -1,
    'excludes_word': -2,
    'mask_index': 0,
    'accept_index': 1,
    'cancel_index': 2,
    'end_index': 3,
    'greet_index': 4,
    'info_question_index': 5,
    'insist_index': 6,
    'slot_question_index': 7,
    'utterance_mask': -1,
    'last_utterance_mask': -2,
    'pad_utterance_mask': -3,
}

KWARGS_DICT = {
    'inference.constrained_beam_search_decoding.ConstrainedBeamSearchDecoding':
        {'num_beams': 3, 'class_rules_mapping': {0: [(7, 1)],
                                                 1: [(8, 1)],
                                                 2: [(6, 1)],
                                                 3: [(0, -1), (1, 1)],
                                                 4: [(9, 1)],
                                                 5: [(2, 1), (4, -1), (5, 1)],
                                                 6: [(10, 1)],
                                                 7: [(3, 1)],
                                                 8: [(11, 1)]}},
    'inference.constrained_gradient_decoding.ConstrainedGradientDecoding':
        {'alpha': 0.1, 'grad_steps': 25},
    'inference.unconstrained_inference.UnconstrainedInference':
        {},
    'learning.constrained_regularized_learning.ConstrainedRegularizedLearning':
        {'epochs': 1},
    'learning.unconstrained_learning.UnconstrainedLearning':
        {'epochs': 1},
    'models.multiwoz_synthetic.psl_model.PSLModelMultiWoZ':
        {'rule_weights': RULE_WEIGHTS, 'rule_names': RULE_NAMES, 'config': DATA_CONFIG},
    'scripts.multiwoz_synthetic.model_util':
        {'input_size': [DATA_CONFIG['max_dialog_size'], DATA_CONFIG['max_utterance_size']]},
}
