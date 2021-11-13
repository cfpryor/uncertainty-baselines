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
import scripts.multiwoz_synthetic.data_util as data_util
import scripts.multiwoz_synthetic.evaluation_util as eval_util
import scripts.multiwoz_synthetic.gradient_decoding_util as gradient_decoding_util
import models.multiwoz_synthetic.psl_model as psl_model

_SEED_RANGE = 10000000


def calculate_metrics(logits, labels, config):
    predictions = tf.math.argmax(tf.concat(logits, axis=0), axis=-1)
    confusion_matrix = eval_util.class_confusion_matrix(predictions, labels, config)
    eval_util.print_metrics(confusion_matrix)


def setup(data_path):
    """Loads data and sets seed."""
    config = gradient_decoding_util.DATA_CONFIG
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
    train_ds, test_ds = data_util.prepare_dataset(data, config)
    logging.info('End: Preparing Dataset')

    logging.info('Building Non-Constrained Model')
    model = gradient_decoding_util.build_model([config['max_dialog_size'], config['max_utterance_size']])

    logging.info('Begin: Non-Constrained Model Learning')
    training_epochs = gradient_decoding_util.TRAINING_EPOCHS
    model.fit(train_ds, epochs=training_epochs)
    logging.info('End: Non-Constrained Model Learning')

    logging.info('Begin: Non-Constrained Model Inference')
    logits = model.predict(test_ds)
    logging.info('End: Non-Constrained Model Inference')

    logging.info('Begin: Non-Constrained Model Analysis')
    calculate_metrics(logits, data['test_truth_dialog'], config)
    logging.info('End: Non-Constrained Model Analysis')

    logging.info('Building Constrained Model')
    rule_weights = gradient_decoding_util.RULE_WEIGHTS
    rule_names = gradient_decoding_util.RULE_NAMES
    constraints = psl_model.PSLModelMultiWoZ(rule_weights, rule_names, config=config)

    logging.info('Begin: Constrained Model Inference')
    alpha = gradient_decoding_util.ALPHA
    grad_steps = gradient_decoding_util.GRAD_STEPS
    inference_application = gradient_decoding.GradientDecoding(model, constraints, alpha=alpha, grad_steps=grad_steps)
    logits = inference_application.predict(test_ds)
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
