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

"""DSTC Synthetic evaluation."""

import tensorflow as tf


def evaluate(predictions, data_path, dataset, config):
    # tf.print(predictions[0], summarize=-1)
    predictions = tf.convert_to_tensor(predictions, dtype=tf.float32)
    confusion_matrix = {index: {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0} for index in range(config['num_labels'])}
    confusion_matrix['total'] = {'correct': 0, 'incorrect': 0}

    for (_, batch_labels, _), batch_predictions in zip(dataset, predictions):
        batch_predictions_index = tf.math.argmax(batch_predictions, axis=-1)
        batch_labels_index = tf.math.argmax(batch_labels, axis=-1)
        confusion_matrix = _batch_confusion_matrix(batch_predictions_index, batch_labels_index, confusion_matrix, config)

    _print_metrics(confusion_matrix)


def _batch_confusion_matrix(predictions, labels, confusion_matrix, config):
    for dialogue_predictions, dialogue_labels in zip(predictions, labels):
        for utterance_prediction, utterance_label in zip(dialogue_predictions, dialogue_labels):
            if utterance_label.numpy() == 0:
                break

            if utterance_prediction == utterance_label:
                confusion_matrix['total']['correct'] += 1
                confusion_matrix[utterance_label.numpy()]['tp'] += 1
            else:
                confusion_matrix['total']['incorrect'] += 1
                confusion_matrix[utterance_label.numpy()]['fp'] += 1
                confusion_matrix[utterance_prediction.numpy()]['fn'] += 1

            for index in range(config['num_labels']):
                if index == utterance_label or index == utterance_prediction:
                    continue
                confusion_matrix[index]['tn'] += 1

    return confusion_matrix


def _precision_recall_f1(confusion_matrix):
    if (confusion_matrix['tp'] + confusion_matrix['fp']) == 0:
        precision = 0.0
    else:
        precision = confusion_matrix['tp'] / (confusion_matrix['tp'] + confusion_matrix['fp'])

    if (confusion_matrix['tp'] + confusion_matrix['fn']) == 0:
        recall = 0.0
    else:
        recall = confusion_matrix['tp'] / (confusion_matrix['tp'] + confusion_matrix['fn'])

    if (precision + recall) == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * (precision * recall / (precision + recall))

    return precision, recall, f1


def _print_metrics(confusion_matrix):
    cat_accuracy = confusion_matrix['total']['correct'] / (
            confusion_matrix['total']['incorrect'] + confusion_matrix['total']['correct'])
    print("Categorical Accuracy: %0.4f" % (cat_accuracy,))
    values = []
    for key, value in confusion_matrix.items():
        if key == 'total':
            continue
        precision, recall, f1 = _precision_recall_f1(value)

        print("Class: %s Precision: %0.4f  Recall: %0.4f  F1: %0.4f" % (str(key).ljust(15), precision, recall, f1))
        values.append(str(precision) + "," + str(recall) + "," + str(f1))
    return values, cat_accuracy
