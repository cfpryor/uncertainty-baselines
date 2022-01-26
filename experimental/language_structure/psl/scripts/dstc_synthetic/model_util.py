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

import tensorflow as tf

def build_model(input_size, learning_rate):
    """Build simple neural model for class prediction."""
    input_layer = tf.keras.layers.Input(input_size)
    hidden_layer_1 = tf.keras.layers.Dense(1024)(input_layer)
    hidden_layer_2 = tf.keras.layers.Dense(
        512, activation='sigmoid')(
        hidden_layer_1)
    output = tf.keras.layers.Dense(
        39, activation='softmax',
        kernel_regularizer=tf.keras.regularizers.l2(1.0))(
        hidden_layer_2)

    model = tf.keras.Model(input_layer, output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model