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

"""Util for the MultiWoZ Synthetic gradient decoding."""

import scripts.multiwoz_synthetic_data_util as data_util

RULE_WEIGHTS = [1.0, 20.0, 5.0, 5.0, 5.0, 10.0, 5.0, 20.0, 5.0, 5.0, 5.0, 10.0]
RULE_NAMES = (
'rule_1', 'rule_2', 'rule_3', 'rule_4', 'rule_5', 'rule_6', 'rule_7', 'rule_8', 'rule_9', 'rule_10', 'rule_11',
'rule_12')

CONFIG = {
    'default_seed': 4,
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
    'shuffle_train': True,
    'shuffle_test': False,
    'train_epochs': 1,
}


def prepare_dataset(data, config):
    """Prepares the train and test datasets."""
    train_dialogs = data_util.add_features(
        data['train_data'],
        vocab_mapping=data['vocab_mapping'],
        accept_words=config['accept_words'],
        cancel_words=config['cancel_words'],
        end_words=config['end_words'],
        greet_words=config['greet_words'],
        info_question_words=config['info_question_words'],
        insist_words=config['insist_words'],
        slot_question_words=config['slot_question_words'],
        includes_word=config['includes_word'],
        excludes_word=config['excludes_word'],
        accept_index=config['accept_index'],
        cancel_index=config['cancel_index'],
        end_index=config['end_index'],
        greet_index=config['greet_index'],
        info_question_index=config['info_question_index'],
        insist_index=config['insist_index'],
        slot_question_index=config['slot_question_index'],
        utterance_mask=config['utterance_mask'],
        pad_utterance_mask=config['pad_utterance_mask'],
        last_utterance_mask=config['last_utterance_mask'],
        mask_index=config['mask_index'])
    train_data = data_util.pad_dialogs(train_dialogs,
                                       config['max_dialog_size'],
                                       config['max_utterance_size'])
    raw_train_labels = data_util.one_hot_string_encoding(
        data['train_truth_dialog'], config['class_map'])
    train_labels = data_util.pad_one_hot_labels(
        raw_train_labels,
        config['max_dialog_size'],
        config['class_map'])
    train_ds = data_util.list_to_dataset(train_data[0],
                                         train_labels[0],
                                         config['shuffle_train'],
                                         config['batch_size'])

    test_dialogs = data_util.add_features(
        data['test_data'],
        vocab_mapping=data['vocab_mapping'],
        accept_words=config['accept_words'],
        cancel_words=config['cancel_words'],
        end_words=config['end_words'],
        greet_words=config['greet_words'],
        info_question_words=config['info_question_words'],
        insist_words=config['insist_words'],
        slot_question_words=config['slot_question_words'],
        includes_word=config['includes_word'],
        excludes_word=config['excludes_word'],
        accept_index=config['accept_index'],
        cancel_index=config['cancel_index'],
        end_index=config['end_index'],
        greet_index=config['greet_index'],
        info_question_index=config['info_question_index'],
        insist_index=config['insist_index'],
        slot_question_index=config['slot_question_index'],
        utterance_mask=config['utterance_mask'],
        pad_utterance_mask=config['pad_utterance_mask'],
        last_utterance_mask=config['last_utterance_mask'],
        mask_index=config['mask_index'])
    test_data = data_util.pad_dialogs(test_dialogs,
                                      config['max_dialog_size'],
                                      config['max_utterance_size'])
    raw_test_labels = data_util.one_hot_string_encoding(
        data['test_truth_dialog'], config['class_map'])
    test_labels = data_util.pad_one_hot_labels(
        raw_test_labels,
        config['max_dialog_size'],
        config['class_map'])
    test_ds = data_util.list_to_dataset(test_data[0], test_labels[0],
                                        config['shuffle_test'],
                                        config['batch_size'])

    return train_ds, test_ds
