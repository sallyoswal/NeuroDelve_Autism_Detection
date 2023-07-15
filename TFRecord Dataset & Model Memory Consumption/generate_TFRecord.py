'''
Functions to iter through the TFRecord datasets
'''
def _parse_function_labels(proto):
    # define your tfrecord again. Remember that you saved your image as a string.
    keys_to_features = {'image': tf.io.FixedLenFeature([], tf.string),
                        "label": tf.io.FixedLenFeature([], tf.int64)}
    
    # Load one example
    parsed_features = tf.io.parse_single_example(proto, keys_to_features)
    
    # Turn your saved image string into an array
    parsed_features['image'] = tf.io.decode_raw(
        parsed_features['image'], tf.float32)
    
    return parsed_features['image'], parsed_features["label"]

  
def create_dataset_labels(filepath):
    
    # This works with arrays as well
    dataset = tf.data.TFRecordDataset(filepath)
    
    # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
    dataset = dataset.map(_parse_function_labels, num_parallel_calls=8)
    dataset = dataset.shuffle(SHUFFLE_BUFFER, reshuffle_each_iteration=True).batch(BATCH_SIZE, drop_remainder = True)
    return dataset
