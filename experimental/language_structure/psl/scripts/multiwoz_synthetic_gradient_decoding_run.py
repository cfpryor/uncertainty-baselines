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

import logging
import os
import random
import sys

import tensorflow as tf

import inference.gradient_decoding as gradient_decoding
import scripts.logs as logs
import scripts.multiwoz_synthetic_data_util as data_util
import scripts.multiwoz_synthetic_gradient_decoding_util as gradient_decoding_util
import scripts.util as util
import models.multiwoz_synthetic.psl_model as psl_model

_SEED_RANGE = 10000000


def build_non_constrained_model(learning_rate, config):
    return build_model([config['max_dialog_size'], config['max_utterance_size']], learning_rate=learning_rate)


def non_constrained_learning(model, train_ds, config):
    model.fit(train_ds, epochs=config['train_epochs'])


def non_constrained_inference(model, test_ds):
    return model.predict(test_ds)


def build_constrained_model(rule_weights, rule_names, config):
    return psl_model.PSLModelMultiWoZ(rule_weights, rule_names, config=config)


def constrained_inference(model, constrained_model, test_ds, alpha, grad_step):
    return gradient_decoding.predict(model, test_ds, constrained_model, alpha=alpha, grad_steps=grad_step)


def calculate_metrics(logits, labels, config):
    predictions = tf.math.argmax(tf.concat(logits, axis=0), axis=-1)
    confusion_matrix = util.class_confusion_matrix(predictions, labels, config)
    util.print_metrics(confusion_matrix)


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

    logging.info('Building Non-Constrained Model')
    model = build_non_constrained_model(0.0001, config)

    logging.info('Begin: Non-Constrained Model Learning')
    non_constrained_learning(model, train_ds, config)
    logging.info('End: Non-Constrained Model Learning')

    logging.info('Begin: Non-Constrained Model Inference')
    logits = non_constrained_inference(model, test_ds)
    logging.info('End: Non-Constrained Model Inference')

    logging.info('Begin: Non-Constrained Model Analysis')
    calculate_metrics(logits, data['test_truth_dialog'], config)
    logging.info('End: Non-Constrained Model Analysis')

    logging.info('Building Constrained Model')
    rule_weights = gradient_decoding_util.RULE_WEIGHTS
    rule_names = gradient_decoding_util.RULE_NAMES
    constrained_model = build_constrained_model(rule_weights, rule_names, config)

    logging.info('Begin: Constrained Model Inference')
    logits = constrained_inference(model, constrained_model, test_ds, 0.1, 25)
    logging.info('End: Constrained Model Inference')

    logging.info('Begin: Constrained Model Analysis')
    calculate_metrics(logits, data['test_truth_dialog'], config)
    logging.info('End: Constrained Model Analysis')


def _load_args(args):
    executable = args.pop(0);
    if len(args) != 1 or ({'h', 'help'} & {arg.lower().strip().replace('-', '') for arg in args}):
        print('USAGE: python3 %s <data path>' % (executable,), file=sys.stderr)
        sys.exit(1)

    return args.pop()


if __name__ == '__main__':
    logs.initLogging()
    main(_load_args(sys.argv))
