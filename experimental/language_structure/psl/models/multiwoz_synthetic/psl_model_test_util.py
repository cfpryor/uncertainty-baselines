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

import tensorflow as tf

# Other Constants
SEED = 4

# Inference Constants
ALPHA = 0.1
GRAD_STEPS = 25

# Learning Constants
TRAINING_EPOCHS = 5
LEARNING_RATE = 0.001

# Data Constants
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


def build_model(input_size, learning_rate=LEARNING_RATE):
    """Build simple neural model for class prediction."""
    input_layer = tf.keras.layers.Input(input_size)
    hidden_layer_1 = tf.keras.layers.Dense(1024)(input_layer)
    hidden_layer_2 = tf.keras.layers.Dense(
        512, activation='sigmoid')(
        hidden_layer_1)
    output = tf.keras.layers.Dense(
        9, activation='softmax',
        kernel_regularizer=tf.keras.regularizers.l2(1.0))(
        hidden_layer_2)

    model = tf.keras.Model(input_layer, output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model

LOGITS = [[[0.0, 0.0, 0.4, 0.4, 0.0, 0.2, 0.0, 0.0, 0.0],
           [0.0, 0.0, 0.2, 0.6, 0.0, 0.2, 0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0],
           [0.0, 0.8, 0.1, 0.1, 0.2, 0.0, 0.0, 0.0, 0.2],
           [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
           [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 1.0],
           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
          [[0.0, 0.8, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0],
           [0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
           [0.0, 0.0, 0.5, 0.4, 0.0, 0.0, 0.0, 0.1, 0.0],
           [0.0, 0.0, 0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
           [0.0, 0.0, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]

FEATURES = [[[-1, -2, -2, -2, -1, -2, -2, -2], [-1, -2, -2, -2, -1, -2, -2, -2],
             [-1, -2, -2, -2, -2, -2, -2, -2], [-1, -2, -2, -2, -2, -1, -2, -1],
             [-1, -2, -2, -1, -1, -2, -2, -2], [-2, -1, -2, -1, -1, -2, -1, -2],
             [-3, -2, -2, -2, -2, -2, -2, -2], [-3, -2, -2, -2, -2, -2, -2, -2],
             [-3, -2, -2, -2, -2, -2, -2, -2], [-3, -2, -2, -2, -2, -2, -2,
                                                -2]],
            [[-1, -2, -2, -2, -2, -2, -2, -2], [-1, -2, -2, -2, -2, -2, -2, -2],
             [-1, -2, -2, -1, -1, -2, -2, -2], [-1, -2, -2, -2, -1, -2, -2, -1],
             [-1, -2, -1, -2, -1, -2, -2, -2], [-2, -2, -2, -1, -1, -2, -2, -2],
             [-3, -2, -2, -2, -2, -2, -2, -2], [-3, -2, -2, -2, -2, -2, -2, -2],
             [-3, -2, -2, -2, -2, -2, -2, -2], [-3, -2, -2, -2, -2, -2, -2,
                                                -2]]]

DATA = {
    'train_data': [
        [[[
            1109, 1616, 41, 800, 740, 1743, 557, 981, 886, 1616, 1658, 909,
            1380, 1256, 1565, 482, 1304
        ], [1109, 1304]],
         [[1109, 1023, 38, 893, 1037, 1664, 886, 1304],
          [
              1109,
              218, 751, 1616, 812, 1406, 1152, 981, 65, 778, 688, 886, 427, 641,
              611, 742, 321, 557, 354, 1471, 161, 182, 767, 1304
          ]],
         [[1109, 1162, 145, 557, 981, 740, 734, 776, 1037, 755, 886, 1304],
          [
              1109, 1616, 812, 1406, 1152, 981, 79, 886, 766, 1616, 558, 165,
              1471, 161, 182, 4, 1304
          ]],
         [[1109, 1738, 145, 893, 532, 1304],
          [
              1109, 1616, 1658, 218, 1616, 812, 1406, 1152, 981, 79, 886, 1023,
              38, 557, 354, 182, 731, 161, 182, 1304
          ]],
         [[1109, 1738, 145, 1215, 1047, 1274, 1304],
          [
              1109, 1616, 812, 1406, 1152, 981, 740, 65, 778, 688, 886, 427,
              641, 611, 742, 321, 557, 354, 1017, 161, 731, 1304
          ]],
         [[1109, 1162, 641, 631, 145, 1738, 1499, 740, 1743, 557, 981, 1304],
          [
              1109, 1616, 1658, 218, 145, 1162, 1499, 981, 740, 263, 173, 62,
              886, 766, 1616, 558, 165, 1471, 161, 1017, 4, 1304
          ]]],
        [[[
            1109, 1616, 1658, 1450, 1743, 800, 1430, 79, 886, 1616, 1658, 1496,
            1565, 1448, 929, 1489, 742, 1662, 1565, 1662, 1304
        ], [1109, 1304]]],
        [[[
            1109, 1616, 1658, 1276, 1450, 1743, 800, 1430, 79, 751, 1616, 1133,
            1431, 1496, 742, 1062, 1415, 1565, 818, 1304
        ], [1109, 1304]]],
        [[[
            1109, 1616, 41, 800, 981, 886, 1616, 1077, 742, 1145, 1565, 83,
            1037, 923, 1304
        ], [1109, 1304]],
         [[1109, 1738, 145, 557, 740, 1743, 557, 981, 909, 256, 680, 187, 1304],
          [
              1109, 218, 1616, 812, 1406, 1152, 981, 740, 886, 1023, 38, 557,
              354, 182, 767, 161, 1017, 4, 1304
          ]],
         [[1109, 525, 641, 751, 1498, 1133, 1431, 1085, 1743, 610, 1304],
          [1109, 427, 641, 611, 742, 865, 641, 557, 574, 1304]],
         [[1109, 525, 641, 751, 1498, 1133, 1431, 1085, 886, 1304],
          [1109, 1185, 641, 1077, 1762, 512, 4, 1304]]],
        [[[
            1109, 764, 1178, 1616, 1658, 1450, 1743, 557, 981, 79, 886, 1616,
            1133, 1431, 1496, 742, 821, 1565, 83, 1304
        ], [1109, 1304]]]
    ],
    'test_data': [
        [[[
            1109, 1616, 1658, 1450, 1743, 891, 38, 800, 1430, 886, 1616, 1658,
            909, 742, 499, 1565, 1159, 1472, 886, 1304
        ], [1109, 1304]]],
        [[[
            1109, 1616, 427, 611, 564, 112, 801, 1412, 742, 446, 248, 800, 1001,
            194, 886, 1616, 1077, 742, 1514, 1743, 142, 886, 1304
        ], [1109, 1304]],
         [[1109, 1738, 1573, 557, 1510, 1561, 1301, 1301, 1412, 4, 1304],
          [
              1109, 1616, 323, 800, 1409, 1177, 886, 1573, 1738, 557, 1412, 742,
              1621, 248, 800, 1001, 194, 886, 1304
          ]],
         [[1109, 1499, 1718, 37, 1738, 1337, 1616, 1077, 886, 1304],
          [
              1109, 800, 1176, 72, 1506, 1738, 1374, 751, 427, 641, 611, 742,
              1514, 1573, 1304
          ]]],
        [[[
            1109, 1228, 1616, 1658, 1450, 1743, 800, 981, 886, 1616, 1077, 742,
            1145, 283, 1669, 1565, 482, 1250, 551, 886, 1304
        ], [1109, 1304]],
         [[1109, 1228, 766, 641, 1406, 1762, 742, 849, 1304],
          [
              1109, 1616, 812, 1406, 1152, 981, 740, 886, 427, 641, 611, 742,
              321, 557, 354, 182, 731, 4, 1304
          ]],
         [[1109, 1718, 37, 1738, 1337, 1616, 1077, 1304],
          [1109, 427, 641, 611, 742, 865, 641, 557, 574, 1304]],
         [[1109, 525, 641, 37, 1738, 1337, 1616, 1077, 886, 1304],
          [1109, 1738, 145, 1762, 512, 1616, 766, 814, 641, 4, 1304]]],
        [[[
            1109, 1228, 1616, 1658, 1450, 1743, 662, 226, 557, 981, 79, 886,
            1616, 1658, 1496, 742, 1187, 1493, 1136, 1565, 1690, 886, 1304
        ], [1109, 1304]]],
    ],
    'vocab_mapping': {
        '?': 4,
        'address': 53,
        'thank': 525,
        'sure': 631,
        'yes': 758,
        'hello': 764,
        'pricey': 1012,
        'hi': 1228,
        'what': 1337,
        'great': 1490,
        'no': 1499,
        'phone': 1596,
        'thanks': 1718,
    },
    'train_truth_dialog': [['init_request', 'second_request', 'second_request',
                            'second_request', 'second_request', 'insist'],
                           ['init_request'],
                           ['init_request'],
                           ['init_request', 'second_request', 'cancel', 'end'],
                           ['init_request']],
    'test_truth_dialog': [['init_request'],
                          ['init_request', 'slot_question', 'cancel'],
                          ['init_request', 'second_request', 'cancel', 'end'],
                          ['init_request']]
}

