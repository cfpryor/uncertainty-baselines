# coding=utf-8
# Copyright 2022 The Uncertainty Baselines Authors.
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

"""PSL utils to load PSL constraint model."""

from typing import Any, Dict, List, Sequence

import tensorflow as tf
import data as data_utils  # local file import from experimental.language_structure.psl
import psl_model  # local file import from experimental.language_structure.psl
import psl_model_multiwoz  # local file import from experimental.language_structure.psl
import data_preprocessor as preprocessor  # local file import from experimental.language_structure.vrnn

_INPUT_ID_NAME = preprocessor.INPUT_ID_NAME


def _check_dataset_supported(dataset: str):
  """Checks if the dataset is supported by PSL constraint model."""
  if dataset != 'multiwoz_synth':
    raise ValueError(
        'Supported PSL constraint for dataset multiwoz_synth, found {}.'.format(
            dataset))


def get_psl_model(dataset: str, rule_names: List[str],
                  rule_weights: List[float], **kwargs) -> psl_model.PSLModel:
  """Constraints PSL constraint model."""
  _check_dataset_supported(dataset)
  return psl_model_multiwoz.PSLModelMultiWoZ(rule_weights, rule_names, **kwargs)


def _get_keyword_ids_per_class(config: Dict[str, Any],
                               vocab: Sequence[str]) -> Sequence[Sequence[int]]:
  """Gets keyword ids for each class in the PSL constraint model."""
  vocab_mapping = {word: word_id for word_id, word in enumerate(vocab)}
  keyword_ids_per_class = []
  for key in [
      'accept_words', 'cancel_words', 'end_words', 'greet_words',
      'info_question_words', 'insist_words', 'slot_question_words'
  ]:
    keyword_ids = [
        vocab_mapping[word] for word in config[key] if word in vocab_mapping
    ]
    if keyword_ids:
      keyword_ids_per_class.append(keyword_ids)
  return keyword_ids_per_class


def _create_psl_features(
    dialogs: tf.Tensor, config: Dict[str, Any],
    keyword_ids_per_class: Sequence[Sequence[int]]) -> tf.Tensor:
  """Creates features for PSL constraint model."""
  features = data_utils.create_features(
      dialogs,
      keyword_ids_per_class,
      include_keyword_value=config['includes_word'],
      exclude_keyword_value=config['excludes_word'],
      pad_utterance_mask_value=config['pad_utterance_mask'],
      utterance_mask_value=config['utterance_mask'],
      last_utterance_mask_value=config['last_utterance_mask'])
  return features


def _create_psl_dialogs(user_utterance_ids: tf.Tensor,
                        system_utterance_ids: tf.Tensor) -> tf.Tensor:
  """Creates dialogs tensor of shape [batch_size, dialog_length, 2, seq_length]."""
  return tf.stack([user_utterance_ids, system_utterance_ids], axis=2)


def psl_feature_mixin(fn: Any, dataset: str, psl_config: Dict[str, Any],
                      vocab: Sequence[str]):
  """Creates PSL feature generation mixin.

  Args:
    fn: dataset processing function converting the dataset into VRNN features.
    dataset: dataset name.
    psl_config: PSL config to create features.
    vocab: vocabulary list.

  Returns:
    decorated `fn` to include PSL input features generation.
  """
  _check_dataset_supported(dataset)
  keyword_ids_per_class = _get_keyword_ids_per_class(psl_config, vocab)

  def _run(inputs: tf.Tensor):
    (input_1, input_2, label_id, label_mask, initial_state, initial_sample,
     domain_label) = inputs
    psl_dialogs = _create_psl_dialogs(input_1[_INPUT_ID_NAME],
                                      input_2[_INPUT_ID_NAME])
    psl_inputs = _create_psl_features(psl_dialogs, psl_config,
                                      keyword_ids_per_class)
    return (input_1, input_2, label_id, label_mask, initial_state,
            initial_sample, domain_label, psl_inputs)

  return lambda inputs: _run(fn(inputs))


def _copy_model_weights(weights: List[tf.Tensor]) -> List[tf.Tensor]:
  """Copies a list of model weights."""
  weights_copy = []
  for layer in weights:
    weights_copy.append(tf.identity(layer))

  return weights_copy


def update_logits(model: tf.keras.Model,
                  optimizer: tf.keras.optimizers.Optimizer, model_inputs: Any,
                  get_logits_fn: Any, psl_constraint: psl_model.PSLModel,
                  psl_inputs: tf.Tensor, grad_steps: int,
                  alpha: float) -> tf.Tensor:
  """Test step for gradient based weight updates.

  Args:
    model: keras model generating the logits
    optimizer: keras optimizer
    model_inputs: model input features
    get_logits_fn: the function deriving the logits from the model outputs.
    psl_constraint: differentable psl constraints
    psl_inputs: psl input features
    grad_steps: number of gradient steps taken to try and satisfy the
      constraints
    alpha: parameter to determine how important it is to keep the constrained
      weights close to the trained unconstrained weights

  Returns:
    Logits after satisfiying constraints.
  """

  @tf.function
  def test_step(model_inputs: Any, psl_inputs: tf.Tensor,
                weights: Sequence[tf.Tensor]):
    """Update weights by satisfing test constraints."""
    with tf.GradientTape() as tape:
      model_outputs = model(model_inputs, training=False)
      logits = get_logits_fn(model_outputs)
      constraint_loss = psl_constraint.compute_loss(psl_inputs, logits)
      weight_loss = tf.reduce_sum([
          tf.reduce_mean(tf.math.squared_difference(w, w_h))
          for w, w_h in zip(weights, model.trainable_weights)
      ])
      loss = constraint_loss + alpha * weight_loss

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  weights_copy = _copy_model_weights(model.trainable_weights)
  for _ in tf.range(tf.cast(grad_steps, dtype=tf.int32)):
    test_step(model_inputs, psl_inputs, weights=weights_copy)

  model_outputs = model(model_inputs)
  logits = get_logits_fn(model_outputs)
  for var, weight in zip(model.trainable_variables, weights_copy):
    var.assign(weight)

  return logits
