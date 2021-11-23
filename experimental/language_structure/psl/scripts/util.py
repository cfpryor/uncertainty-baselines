import importlib
import json

import tensorflow as tf


def qualified_import(qualified_name):
    """
    Import a fully qualified name, e.g. 'psl.scripts.util.qualified_import'.
    """

    if qualified_name is None or qualified_name == '' or qualified_name == 0:
        raise ValueError("Empty name supplied for import.")

    parts = qualified_name.split('.')
    module_name = '.'.join(parts[0:-1])
    target_name = parts[-1]

    if len(parts) == 1:
        raise ValueError("Non-qualified name supplied for import: " + qualified_name)

    try:
        module = importlib.import_module(module_name)
    except ImportError:
        raise ValueError("Unable to locate module (%s) for qualified object (%s)." %
                         (module_name, qualified_name))

    if target_name == '':
        return module

    return getattr(module, target_name)


def list_to_dataset(data, labels, shuffle: bool, batch_size: int) -> tf.data.Dataset:
    """Converts list into tensorflow dataset."""
    ds = tf.data.Dataset.from_tensor_slices((data, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(data))
    ds = ds.batch(batch_size)
    return ds


def load_json(path: str):
    with tf.io.gfile.GFile(path, 'r') as json_file:
        return json.load(json_file)
