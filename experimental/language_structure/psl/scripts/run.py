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

import scripts.logs as logs
import scripts.util as util

_SEED_RANGE = 10000000


def setup(data_path, experiment_name, constraints_name, learning_name, inference_name):
    global constants
    global data_util
    global eval_util
    global model_util

    constants = util.qualified_import('scripts.' + experiment_name + '.constants.')
    data_util = util.qualified_import('scripts.' + experiment_name + '.data_util.')
    eval_util = util.qualified_import('scripts.' + experiment_name + '.eval_util.')
    model_util = util.qualified_import('scripts.' + experiment_name + '.model_util.')
    constraints_class = util.qualified_import(constraints_name)
    learning_class = util.qualified_import(learning_name)
    inference_class = util.qualified_import(inference_name)

    config = constants.DATA_CONFIG
    seed = random.randint(-_SEED_RANGE, _SEED_RANGE)
    logging.info('Seed: %d' % (seed,))
    tf.random.set_seed(seed)

    if not os.path.exists(data_path):
        raise FileNotFoundError('%s' % (data_path,))
    data = util.load_json(data_path)

    return data, config, constraints_class, learning_class, inference_class


def main(data_path, experiment_name, constraints_name, learning_name, inference_name):
    logging.info('Begin: Loading Data')
    data, config, constraints_class, learning_class, inference_class = setup(data_path, experiment_name, constraints_name, learning_name, inference_name)
    train_ds, test_ds = data_util.prepare_dataset(data, config)
    logging.info('End: Loading Data')

    logging.info('Begin: Building Neural Model and Constraints')
    neural_model_kwargs = constants.KWARGS_DICT['scripts.' + experiment_name + '.model_util']
    neural_model = model_util.build_model(**neural_model_kwargs)

    constraints_kwargs = constants.KWARGS_DICT[constraints_name]
    constraints = constraints_class(**constraints_kwargs)
    logging.info('End: Building Model and Constraints')

    logging.info('Begin: Learning -- %s' % (learning_name,))
    learning_kwargs = constants.KWARGS_DICT[learning_name]
    learning = learning_class(neural_model, constraints, **learning_kwargs)
    learning.fit(train_ds)
    logging.info('End: Learning-- %s' % (learning_name,))

    logging.info('Begin: Inference-- %s' % (inference_name,))
    inference_kwargs = constants.KWARGS_DICT[inference_name]
    inference = inference_class(neural_model, constraints, **inference_kwargs)
    output = inference.predict(test_ds)
    logging.info('End: Inference -- %s' % (inference_name,))

    logging.info('Begin: Evaluation')
    eval_util.evaluate(output, data, config)
    logging.info('End: Evaluation')


def _load_args(args):
    executable = args.pop(0);
    if len(args) != 5 or ({'h', 'help'} & {arg.lower().strip().replace('-', '') for arg in args}):
        print('USAGE: python3 %s <data path> <experiment> <constraints module> <learning module> <inference module>' % (executable,), file=sys.stderr)
        sys.exit(1)

    data_path = args.pop(0)
    experiment_name = args.pop(0)
    constraints_name = args.pop(0)
    learning_name = args.pop(0)
    inference_name = args.pop(0)

    return data_path, experiment_name, constraints_name, learning_name, inference_name


if __name__ == '__main__':
    logs.initLogging()
    data_path, experiment_name, constraints_name, learning_name, inference_name = _load_args(sys.argv)
    main(data_path, experiment_name, constraints_name, learning_name, inference_name)
