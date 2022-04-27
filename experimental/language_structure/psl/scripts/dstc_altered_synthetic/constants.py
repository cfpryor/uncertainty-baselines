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

"""DSTC Altered Synthetic constants."""

RULE_WEIGHTS = [1.0, 1.0]
RULE_NAMES = ('rule_1', 'rule_2')

DATA_CONFIG = {
    'num_batches': 50,
    'batch_size': 16,
    'max_dialog_size': 24,
    'max_utterance_size': 76,
    'num_labels': 39,
}

MODEL_CONFIG = {
    'learning_rate': 1e-2,
    'loss': 'categorical_crossentropy'
}

KWARGS_DICT = {
    'inference.constrained_gradient_decoding.ConstrainedGradientDecoding':
        {'alpha': 1000000000, 'grad_steps': 100},
    'inference.unconstrained_inference.UnconstrainedInference':
        {},
    'learning.constrained_regularized_learning.ConstrainedRegularizedLearning':
        {'epochs': 10},
    'learning.unconstrained_learning.UnconstrainedLearning':
        {'epochs': 7},
    'models.dstc_altered_synthetic.psl_model.PSLModelDSTCAlteredSynthetic':
        {'rule_weights': RULE_WEIGHTS, 'rule_names': RULE_NAMES, 'config': DATA_CONFIG},
    'scripts.dstc_altered_synthetic.model_util':
        {'input_size': [DATA_CONFIG['max_dialog_size'], 2, DATA_CONFIG['max_utterance_size']]},
}
