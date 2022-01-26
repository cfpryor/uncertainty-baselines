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
"""MultiWoZ Synthetic data manipulation."""

import copy
import numpy as np
from typing import Dict, List, Tuple

import scripts.util as util

Utterance = List[int]
UserSystem = List[Utterance]
Dialog = List[UserSystem]


def prepare_dataset(data_path, config, data=None):
    """Prepares the train and test datasets."""
    if data is None:
        data = util.load_json(data_path)
    train_ds = _prepare_dataset_helper(data['train_data'],
                                       data['train_truth_dialog'],
                                       data['vocab_mapping'],
                                       True, config)

    test_ds = _prepare_dataset_helper(data['test_data'],
                                      data['test_truth_dialog'],
                                      data['vocab_mapping'],
                                      False, config)

    return train_ds, test_ds


def _prepare_dataset_helper(raw_data, raw_labels, vocab_mapping, training, config):
    data = _pad_dialogs(raw_data,
                        config['max_dialog_size'],
                        config['max_utterance_size'])
    psl_features = _create_psl_features(
        data[0],
        vocab_mapping=vocab_mapping,
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
    labels = _one_hot_string_encoding(raw_labels,
                                      config['class_map'])
    labels = _pad_one_hot_labels(labels,
                                 config['max_dialog_size'],
                                 config['class_map'])
    return util.list_to_dataset(data[0], labels[0], psl_features, training, config['batch_size'])


def _annotate_if_contains_words(features: List[int], utterance: List[int],
                                key_words: List[str], vocab_mapping: Dict[str, int],
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
      A utterance of length max utterance size annotated with weither it
      includes/excludes at least one key word.
    """
    features[word_index] = excludes_word
    for word in key_words:
        if vocab_mapping[word] in utterance:
            features[word_index] = includes_word
            break

    return features


def _create_psl_features(dialogs: List[Dialog], vocab_mapping: Dict[str, int],
                         accept_words: List[str], cancel_words: List[str],
                         end_words: List[str], greet_words: List[str],
                         info_question_words: List[str], insist_words: List[str],
                         slot_question_words: List[str], includes_word: int,
                         excludes_word: int, accept_index: int, cancel_index: int,
                         end_index: int, greet_index: int, info_question_index: int,
                         insist_index: int, slot_question_index: int,
                         utterance_mask: int, pad_utterance_mask: int,
                         last_utterance_mask: int, mask_index: int) -> List[Dialog]:
    """Annotates if it contains any special tokens.

    This function adds the following features:
      - Padding Mask (indicator if an utterance is a padding)
      - Has Greet Word (indicator if an utterance contains a known greet word)
      - Has End Word (indicator if an utterance contains a known end word)

    Args:
      dialogs: list of dialogs being annotated with special tokens.
      vocab_mapping: a dictionary that maps a vocab to integers.
      accept_words: list of strings representing the known accept words.
      cancel_words: list of strings representing the known cancel words.
      end_words: list of strings representing the known end words.
      greet_words: list of strings representing the known greet words.
      info_question_words: list of strings representing the known info question
        words.
      insist_words: list of strings representing the known insist words.
      slot_question_words: list of strings representing the known slot question
        words.
      includes_word: an integer indicating an utterance does contain a key word.
      excludes_word: an integer indicating an utterance does not contain a key
        word.
      accept_index: an integer representing the index an accept annotation will be
        placed in the utterance.
      cancel_index: an integer representing the index a cancel annotation will be
        placed in the utterance.
      end_index: an integer representing the index an end annotation will be
        placed in the utterance.
      greet_index: an integer representing the index a greet annotation will be
        placed in the utterance.
      info_question_index: an integer representing the index an info question
        annotation will be placed in the utterance.
      insist_index: an integer representing the index an insist annotation will be
        placed in the utterance.
      slot_question_index: an integer representing the index a slot question
        annotation will be placed in the utterance.
      utterance_mask: an integer indicating if it is not a padded utterance.
      pad_utterance_mask: an integer indicating if it is a padded utterance.
      last_utterance_mask: an integer indicating if it is the final utterance
        before padding.
      mask_index: an integer representing the index the utterance mask will be
        placed in the utterance.

    Returns:
      A copy of dialogs with annotations for special tokens included.
    """
    features = []
    for index_i in range(len(dialogs)):
        first_padding = True
        features.append([])
        for index_j in range(len(dialogs[index_i])):
            # Add default values for features.
            utt_features = [0] * 8
            utterance = dialogs[index_i][index_j]

            # Checks if the utterance in dialog is a padded utterance.
            utt_features[mask_index] = utterance_mask
            if all(word == 0 for word in utterance):
                utt_features[mask_index] = pad_utterance_mask

                # Checks if this is the first padding.
                if first_padding and index_j != 0:
                    # Sets previous statement to last utterance in dialog.
                    features[index_i][index_j - 1][mask_index] = last_utterance_mask
                    first_padding = False
            # Check edge case where last utterance is not padding.
            elif first_padding and index_j == (len(dialogs[index_i]) - 1):
                utt_features[mask_index] = last_utterance_mask

            # Checks if utterance in dialog contains a known accept word.
            utt_features = _annotate_if_contains_words(utt_features, utterance,
                                                       accept_words, vocab_mapping, accept_index,
                                                       excludes_word, includes_word)

            # Checks if utterance in dialog contains a known cancel word.
            utt_features = _annotate_if_contains_words(utt_features, utterance,
                                                       cancel_words, vocab_mapping, cancel_index,
                                                       excludes_word, includes_word)

            # Checks if utterance in dialog contains a known end word.
            utt_features = _annotate_if_contains_words(utt_features, utterance,
                                                       end_words, vocab_mapping, end_index,
                                                       excludes_word, includes_word)

            # Checks if utterance in dialog contains a known greet word.
            utt_features = _annotate_if_contains_words(utt_features, utterance,
                                                       greet_words, vocab_mapping, greet_index,
                                                       excludes_word, includes_word)

            # Checks if utterance in dialog contains a known info question word.
            utt_features = _annotate_if_contains_words(utt_features, utterance,
                                                       info_question_words,
                                                       vocab_mapping, info_question_index,
                                                       excludes_word, includes_word)

            # Checks if utterance in dialog contains a known insist word.
            utt_features = _annotate_if_contains_words(utt_features, utterance,
                                                       insist_words, vocab_mapping, insist_index,
                                                       excludes_word, includes_word)

            # Checks if utterance in dialog contains a known slot question word.
            utt_features = _annotate_if_contains_words(utt_features, utterance,
                                                       slot_question_words, vocab_mapping,
                                                       slot_question_index,
                                                       excludes_word, includes_word)

            # Sets utterance with new features.
            features[index_i].append(utt_features)

    return features


def _pad_utterance(utterance: Utterance,
                   max_utterance_size: int) -> Tuple[Utterance, List[int]]:
    """Pads utterance up to the max utterance size."""
    utt = utterance + [0] * (max_utterance_size - len(utterance))
    mask = [1] * len(utterance) + [0] * (max_utterance_size - len(utterance))
    return utt, mask


def _pad_dialog(
        dialog: Dialog, max_dialog_size: int, max_utterance_size: int) -> Tuple[List[List[int]], List[List[int]]]:
    """Pads utterances in a dialog up to max dialog sizes."""

    dialog_usr_input, dialog_sys_input = [], []

    for turn in dialog:
        pad_utt, mask = _pad_utterance(turn[0], max_utterance_size)
        dialog_usr_input.append(pad_utt)

        pad_utt, mask = _pad_utterance(turn[1], max_utterance_size)
        dialog_sys_input.append(pad_utt)

    for _ in range(max_dialog_size - len(dialog)):
        pad_utt, mask = _pad_utterance([], max_utterance_size)
        dialog_usr_input.append(pad_utt)

        pad_utt, mask = _pad_utterance([], max_utterance_size)
        dialog_sys_input.append(pad_utt)

    return dialog_usr_input, dialog_sys_input


def _pad_dialogs(
        dialogs: List[Dialog], max_dialog_size: int, max_utterance_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Pads all dialogs and utterances."""
    usr_input_sent, sys_input_sent = [], []

    for dialog in dialogs:
        usr_input, sys_input = _pad_dialog(dialog, max_dialog_size, max_utterance_size)

        usr_input_sent.append(usr_input)
        sys_input_sent.append(sys_input)

    return np.array(usr_input_sent), np.array(sys_input_sent)


def _one_hot_string_encoding(labels: List[List[str]],
                             mapping: Dict[str, int]) -> List[List[List[int]]]:
    """Converts string labels into one hot encodings."""
    one_hot_labels = []

    for dialog in labels:
        one_hot_labels.append([])
        for utterance in dialog:
            one_hot_labels[-1].append([0] * len(mapping))
            one_hot_labels[-1][-1][mapping[utterance]] = 1

    return one_hot_labels


def _pad_one_hot_labels(
        labels: List[List[List[int]]], max_dialog_size: int,
        mapping: Dict[str, int]) -> Tuple[List[List[List[int]]], List[List[int]]]:
    """Pads one hot encoded labels."""
    pad_labels = []
    pad_mask = []

    for dialog in labels:
        pad_labels.append(dialog)
        pad_mask.append([1] * len(dialog) + [0] * (max_dialog_size - len(dialog)))

        for _ in range(max_dialog_size - len(dialog)):
            pad_labels[-1].append([0] * len(mapping))

    return pad_labels, pad_mask
