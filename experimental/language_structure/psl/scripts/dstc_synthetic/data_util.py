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

from typing import List

import scripts.util as util
import uncertainty_baselines.datasets.dialog_state_tracking as data_loader


def prepare_dataset(data_dir, config):
    """Prepares the train and test datasets."""
    train_data_loader = data_loader.SGDSynthDataset(data_dir, split="train")
    train_ds = train_data_loader.load(batch_size=config['batch_size'])
    train_ds = _create_dataset(train_ds, config, True)

    test_data_loader = data_loader.SGDSynthDataset(data_dir, split="test")
    test_ds = test_data_loader.load(batch_size=config['batch_size'])
    test_ds = _create_dataset(test_ds, config, False)
    return train_ds, test_ds

def _create_dataset(dataset, config, training):
    current_batch = 0
    data = []
    labels = []
    psl_data = []
    for batch in dataset:
        # TODO(connor) - temporary, remove check for real dataset.
        if current_batch == config['num_batches']:
            break

        for usr_dialogue, sys_dialogue, raw_usr_dialogue, raw_sys_dialogue, dialogue_labels in zip(batch['usr_utt'], batch['sys_utt'], batch['usr_utt_raw'], batch['sys_utt_raw'], batch['label']):
            # TODO(connor) temporary, add system dialogues to neural model.
            data.append(usr_dialogue.numpy())
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
    first_padding = True
    features = []
    for index_i in range(len(dialogue[0])):
        # Add default values for features.
        features.append([0] * 5)

        # Checks if the utterance in dialog is a padded utterance.
        features[-1][config['mask_index']] = config['utterance_mask']
        if dialogue[0][index_i].numpy() == b'' and dialogue[1][index_i].numpy() == b'':
            # print(dialogue[0][index_i].numpy(), dialogue[1][index_i])
            features[-1][config['mask_index']] = config['pad_utterance_mask']

            # Checks if this is the first padding.
            if first_padding:
                # Sets previous statement to last utterance in dialog.
                features[index_i - 1][config['mask_index']] = config['last_utterance_mask']
                first_padding = False
        # Check edge case where last utterance is not padding.
        elif first_padding and index_i == (len(dialogue[0]) - 1):
            features[index_i - 1][config['mask_index']] = config['last_utterance_mask']

        for key, value in config['words'].items():
            features[-1] = _annotate_if_contains_words(features[-1], dialogue[0][index_i].numpy().split(),
                                                       value['usr']['words'], value['usr']['index'],
                                                       config['excludes_word'], config['includes_word'])

            features[-1] = _annotate_if_contains_words(features[-1], dialogue[1][index_i].numpy().split(),
                                                       value['sys']['words'], value['sys']['index'],
                                                       config['excludes_word'], config['includes_word'])

    return features

def _annotate_if_contains_words(features: List[int], utterance: List[int],
                                key_words: List[str],
                                word_index: int, excludes_word: int,
                                includes_word: int) -> List[int]:
    """Annotates an utterance if it contains at least one word from a list.

    Args:
      features: list of psl features.
      utterance: list of integers of length max utterance size representing a
        sentence.
      key_words: list of strings representing the words being looked for.
      vocab_mapping: a dictionary that maps vocab to integers.
      word_index: an integer representing the index an annotation will be placed
        in the utterance.
      excludes_word: an integer indicating an utterance does not contain any key
        words.
      includes_word: an integer indicating an utterance does contain a key word.

    Returns:
      A utterance of length max utterance size annotated with whether it
      includes/excludes at least one key word.
    """
    features[word_index] = excludes_word
    for word in key_words:
        if bytes(word, 'utf-8') in utterance:
            features[word_index] = includes_word
            break

    return features

def analyze_data(dataset, config):
    current_batch = 0
    usr_label_dict = {}
    sys_label_dict = {}
    state_transitions_dict = {0: {}, -1: {}}

    last_label = -1
    for batch in dataset:
        if current_batch == config['num_batches']:
            break

        for usr_dialogue, sys_dialogue, labels in zip(batch['usr_utt_raw'], batch['sys_utt_raw'], batch['label']):
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

        current_batch += 1

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