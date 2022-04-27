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
"""Util file for psl rules test."""

# (words: 2, classes: 4)
WEIGHTS = [[10.0, 20.0, 30.0, 40.0],
           [50.0, 60.0, 70.0, 80.0]]

# (dialogues: 2, utterances: 3, classes: 4)
LOGITS = [[[0.0, 0.0, 0.5, 0.5],
           [1.0, 0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0, 0.0]],
          [[0.2, 0.8, 0.0, 0.0],
           [0.0, 0.3, 0.7, 0.0],
           [0.0, 0.0, 0.0, 0.0]]]

# (dialogues: 2, utterances: 3, words: 2)
FEATURES = [[[1, 0], [0, 1], [0, 0]],
            [[0, 0], [1, 0], [0, 0]]]

DATA_CONFIG = {
    'num_batches': 1,
    'batch_size': 2,
    'max_dialog_size': 3,
    'max_utterance_size': 4,
    'num_labels': 39,
    'pos_psl_weights': WEIGHTS
}


