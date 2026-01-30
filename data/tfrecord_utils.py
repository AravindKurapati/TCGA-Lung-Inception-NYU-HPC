"""TFRecord creation and parsing utilities"""

import tensorflow as tf

def _bytes_feature(value):
    """Create bytes feature for TFRecord"""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(value).numpy()]))

def _int64_feature(value):
    """Create int64 feature for TFRecord"""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))

def serialize_example(image, label):
    """Serialize image and label to TFRecord example"""
    image = tf.cast(image, tf.float32)
    feature = {
        'image': _bytes_feature(image),
        'label': _int64_feature(label)
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def write_tfrecords(filename, images, labels, data_augmentation=None):
    """Write images and labels to TFRecord file"""
    with tf.io.TFRecordWriter(filename) as writer:
        for image, label in zip(images, labels):
            if data_augmentation:
                augmented_img = data_augmentation(tf.expand_dims(image, axis=0))
                tf_example = serialize_example(augmented_img[0], label)
            else:
                tf_example = serialize_example(image, label)
            writer.write(tf_example)
    print(f"TFRecord saved: {filename}")

def create_tfrecord_dataset(tfrecord_path, batch_size, img_size, is_training=True):
    """Create TensorFlow dataset from TFRecord file"""
    def parse_example(example_proto):
        features = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64)
        }
        parsed_features = tf.io.parse_single_example(example_proto, features)
        image = tf.io.parse_tensor(parsed_features['image'], out_type=tf.float32)
        label = tf.cast(parsed_features['label'], tf.int32)
        image = tf.reshape(image, [img_size, img_size, 128])
        return image, label
    
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_example)
    
    if is_training:
        dataset = dataset.shuffle(1000).batch(batch_size).repeat()
    else:
        dataset = dataset.batch(batch_size)
    
    return dataset
