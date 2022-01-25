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
"""DSCT Synthetic data manipulation."""

import logging
import os
import sys

import scripts.logs as logs
import scripts.util as util
import uncertainty_baselines.datasets.dialog_state_tracking as data_loader

import tensorflow as tf

def prepare_dataset(data_dir, config):
    """Prepares the train and test datasets."""
    train_data_loader = data_loader.SGDSynthDataset(data_dir, split="train")
    # train_ds = train_data_loader.load(batch_size=32768)
    train_ds = train_data_loader.load(batch_size=1024)
    # train_ds = _create_psl_features(train_ds, config)
    analyze_data(train_ds)

    test_data_loader = data_loader.SGDSynthDataset(data_dir, split="test")
    # test_ds = test_data_loader.load(batch_size=32768)
    test_ds = test_data_loader.load(batch_size=1024)
    return train_ds, test_ds

def _create_psl_features(dataset, config):
    current_batch = 0
    for batch in dataset:
        if current_batch == config['num_batches']:
            break

        batch['psl_features'] = {'a', 1}
        current_batch += 1
        break


def analyze_data(dataset):
    usr_label_dict = {}
    sys_label_dict = {}
    state_transitions_dict = {0: {}, -1: {}}

    last_label = -1
    for batch in dataset:
        for usr_dialogue, sys_dialogue, labels in zip(batch['usr_utt_raw'], batch['sys_utt_raw'], batch['label']):
            print(usr_dialogue)
            print(sys_dialogue)
            return
            if labels[0].numpy() not in state_transitions_dict[0]:
                state_transitions_dict[0][labels[0].numpy()] = 0
            state_transitions_dict[0][labels[0].numpy()] += 1

            for usr_utt, sys_utt, label in zip(usr_dialogue, sys_dialogue, labels):
                if label == 0:
                    if last_label not in state_transitions_dict[-1]:
                        state_transitions_dict[-1][last_label] = 0
                    state_transitions_dict[-1][last_label] += 1
                    break
                if label.numpy() not in state_transitions_dict:
                    state_transitions_dict[label.numpy()] = {}

                if last_label not in state_transitions_dict[label.numpy()]:
                    state_transitions_dict[label.numpy()][last_label] = 0
                state_transitions_dict[label.numpy()][last_label] += 1
                last_label = label.numpy()

                if label.numpy() not in usr_label_dict:
                    usr_label_dict[label.numpy()] = {}
                for usr_word in usr_utt.numpy().split():
                    if usr_word not in usr_label_dict[label.numpy()]:
                        usr_label_dict[label.numpy()][usr_word] = 0
                    usr_label_dict[label.numpy()][usr_word] += 1

                if label.numpy() not in sys_label_dict:
                    sys_label_dict[label.numpy()] = {}
                for sys_word in sys_utt.numpy().split():
                    if sys_word not in sys_label_dict[label.numpy()]:
                        sys_label_dict[label.numpy()][sys_word] = 0
                    sys_label_dict[label.numpy()][sys_word] += 1
        break

    for label_name, transition_dict in sorted(state_transitions_dict.items()):
        print("\n\nUser State: %d\n" % (label_name,))
        sorted_transition_dict = sorted(transition_dict, key=transition_dict.get, reverse=True)[:5]
        for transition in sorted_transition_dict:
            print("Previous State: %s Count: %d" % (transition, transition_dict[transition]))

    for label_name, word_dict in sorted(usr_label_dict.items()):
        print("\n\nSystem State: %d\n" % (label_name,))
        sorted_word_dict = sorted(word_dict, key=word_dict.get, reverse=True)[:50]
        for word in sorted_word_dict:
            print("Word: %s Count: %d" % (word, word_dict[word]))

    for label_name, word_dict in sorted(sys_label_dict.items()):
        print("\n\nState: %d\n" % (label_name,))
        sorted_word_dict = sorted(word_dict, key=word_dict.get, reverse=True)[:50]
        for word in sorted_word_dict:
            print("Word: %s Count: %d" % (word, word_dict[word]))