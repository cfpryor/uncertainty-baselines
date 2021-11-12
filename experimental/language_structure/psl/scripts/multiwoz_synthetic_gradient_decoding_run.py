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

import json
import logging
import os
import random
import sys

import numpy as np
import tensorflow as tf

import inference.gradient_decoding as inference_application
import scripts.logs as logs
import scripts.multiwoz_synthetic_data_util as data_util
import scripts.multiwoz_synthetic_gradient_decoding_util as gradient_decoding_util
import scripts.util as util
import models.multiwoz_synthetic.psl_model as psl_model

_SEED_RANGE = 10000000


def non_constrained_learning(train_ds, learning_rate, config):
    model = build_model([config['max_dialog_size'], config['max_utterance_size']], learning_rate=learning_rate)
    model.fit(train_ds, epochs=config['train_epochs'])

    return model


def non_constrained_inference(model, test_ds, test_labels, config):
    logits = model.predict(test_ds)
    predictions = tf.math.argmax(logits, axis=-1)

    confusion_matrix = util.class_confusion_matrix(predictions, test_labels, config)
    metrics, cat_accuracy = util.print_metrics(confusion_matrix)

    return model, metrics, cat_accuracy


def build_model(input_size, learning_rate=0.001):
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


def setup(data_path):
    """Loads data and sets seed."""
    config = gradient_decoding_util.CONFIG
    seed = random.randint(-_SEED_RANGE, _SEED_RANGE)
    logging.info('Seed: %d' % (seed,))
    tf.random.set_seed(seed)

    if not os.path.exists(data_path):
        raise FileNotFoundError('%s' % (data_path,))

    logging.info('Begin: Loading Data -- %s' % (data_path))
    data = data_util.load_json(data_path)
    logging.info('End: Loading Data -- %s' % (data_path))

    return data, config


def main(data_path):
    """Runs MultiWoZ experiment."""
    data, config = setup(data_path)

    logging.info('Begin: Preparing Dataset')
    train_ds, test_ds = gradient_decoding_util.prepare_dataset(data, config)
    logging.info('End: Preparing Dataset')

    logging.info('Begin: Non-Constrained Model Learning')
    model = non_constrained_learning(train_ds, 0.0001, config)
    logging.info('End: Non-Constrained Model Learning')

    logging.info('Begin: Non-Constrained Model Inference')
    model, metrics, cat = non_constrained_inference(model, test_ds, data['train_truth_dialog'], config)
    logging.info('End: Non-Constrained Model Inference')


def _load_args(args):
    executable = args.pop(0);
    if len(args) != 1 or ({'h', 'help'} & {arg.lower().strip().replace('-', '') for arg in args}):
        print('USAGE: python3 %s <data path>' % (executable,), file=sys.stderr)
        sys.exit(1)

    return args.pop()


if __name__ == '__main__':
    logs.initLogging()
    main(_load_args(sys.argv))
