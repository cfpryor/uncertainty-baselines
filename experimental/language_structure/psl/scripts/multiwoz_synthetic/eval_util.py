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

"""MultiWoZ Synthetic evaluation."""

import tensorflow as tf

import scripts.util as util


def evaluate(predictions, data_path, config):
    data = util.load_json(data_path)
    labels = data['test_truth_dialog']
    predictions = tf.math.argmax(tf.concat(predictions, axis=0), axis=-1)
    confusion_matrix = _class_confusion_matrix(predictions, labels, config)
    _print_metrics(confusion_matrix)


def _class_confusion_matrix(preds, labels, config):
    class_map = config['class_map']
    reverse_class_map = {v: k for k, v in class_map.items()}
    class_confusion_matrix_dict = {key: {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0} for key, _ in class_map.items()}
    class_confusion_matrix_dict['total'] = {'correct': 0, 'incorrect': 0}

    for pred_list, label_list in zip(preds, labels):
        for pred, label in zip(pred_list, label_list):
            if class_map[label] == pred:
                class_confusion_matrix_dict['total']['correct'] += 1
                class_confusion_matrix_dict[label]['tp'] += 1
            else:
                class_confusion_matrix_dict['total']['incorrect'] += 1
                class_confusion_matrix_dict[label]['fp'] += 1
                class_confusion_matrix_dict[reverse_class_map[pred.numpy()]]['fn'] += 1

            for key in class_map:
                if key == label or key == reverse_class_map[pred.numpy()]:
                    continue
                class_confusion_matrix_dict[reverse_class_map[pred.numpy()]]['tn'] += 1

    return class_confusion_matrix_dict


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

        print("Class: %s Precision: %0.4f  Recall: %0.4f  F1: %0.4f" % (key.ljust(15), precision, recall, f1))
        values.append(str(precision) + "," + str(recall) + "," + str(f1))
    return values, cat_accuracy
