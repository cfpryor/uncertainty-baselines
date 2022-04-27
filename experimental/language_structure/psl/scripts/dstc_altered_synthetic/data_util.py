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
"""DSCT Altered Synthetic data manipulation."""
import os

import scripts.util as util
import uncertainty_baselines.datasets.dialog_state_tracking as data_loader

import numpy as np
import tensorflow as tf

_PSL_TRAIN_DIR = 'psl_train'
_PSL_TEST_DIR = 'psl_test'

_WEIGHTS_FILENAME = 'word_weights_supervised_full.npy'
_VOCAB_TO_ID_FILENAME = 'vocab_to_id.json'


def prepare_dataset(data_dir, config):
    """Prepares the train and test datasets."""
    config['psl_weights'] = np.load(os.path.join(data_dir, _WEIGHTS_FILENAME))
    config['pos_psl_weights'] = config['psl_weights'] * (config['psl_weights'] > 0)
    config['neg_psl_weights'] = (-1 * config['psl_weights']) * (-1 * config['psl_weights'] > 0)
    config['vocab_to_id'] = util.load_json(os.path.join(data_dir, _VOCAB_TO_ID_FILENAME))

    if os.path.isdir(os.path.join(data_dir, 'psl_train')):
        print('Data Found -- Loading: %s' % (os.path.join(data_dir, _PSL_TRAIN_DIR)))
        train_ds = tf.data.experimental.load(os.path.join(data_dir, _PSL_TRAIN_DIR))
        test_ds = tf.data.experimental.load(os.path.join(data_dir, _PSL_TEST_DIR))
        return train_ds, test_ds

    train_data_loader = data_loader.SGDSynthDataset(data_dir, split="train")
    train_ds = train_data_loader.load(batch_size=config['batch_size'])
    train_ds = _create_dataset(train_ds, config, True)
    tf.data.experimental.save(train_ds, os.path.join(data_dir, _PSL_TRAIN_DIR))

    test_data_loader = data_loader.SGDSynthDataset(data_dir, split="test")
    test_ds = test_data_loader.load(batch_size=config['batch_size'])
    test_ds = _create_dataset(test_ds, config, False)
    tf.data.experimental.save(test_ds, os.path.join(data_dir, _PSL_TEST_DIR))

    return train_ds, test_ds


def _create_dataset(dataset, config, training):
    current_batch = 0
    data = []
    labels = []
    psl_data = []
    for batch in dataset:
        print("Current Batch: %d" % (current_batch,))
        if current_batch == config['num_batches']:
            break

        for usr_dialogue, sys_dialogue, raw_usr_dialogue, raw_sys_dialogue, dialogue_labels in zip(batch['usr_utt'], batch['sys_utt'], batch['usr_utt_raw'], batch['sys_utt_raw'], batch['label']):
            data.append([])
            for usr_utt, sys_utt in zip(usr_dialogue.numpy(), sys_dialogue.numpy()):
                data[-1].append([usr_utt, sys_utt])
            labels.append(_one_hot_string_encoding(dialogue_labels.numpy(), config['num_labels']))
            psl_data.append(_create_psl_features([raw_usr_dialogue, raw_sys_dialogue], config))

        current_batch += 1

    return util.list_to_dataset(data, labels, psl_data, training, config['batch_size'])


def _one_hot_string_encoding(labels, num_labels):
    """Converts string labels into one hot encodings."""
    one_hot_labels = []
    for label in labels:
        one_hot_labels.append([0] * num_labels)
        one_hot_labels[-1][label] = 1

    return one_hot_labels


def _create_psl_features(dialogue, config):
    """Creates additional data needed for PSL."""
    features = []

    for index in range(len(dialogue[0])):
        features.append([0] * len(config['vocab_to_id']))
        for utterance in [dialogue[0][index], dialogue[1][index]]:
            if utterance.numpy() == b'':
                continue

            for word in utterance.numpy().split():
                if word.decode('utf-8') not in config['vocab_to_id']:
                    continue
                features[-1][config['vocab_to_id'][word.decode('utf-8')]] = 1

    return features
