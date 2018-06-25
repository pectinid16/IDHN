import tensorflow as tf


def read_and_decode(filename, epochs,size):
    filename_queue = tf.train.string_input_producer([filename],num_epochs=epochs)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([38], tf.float32),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.int16)
    img = tf.reshape(img, [size, size, 3])
    img = tf.cast(img, tf.float32)
    label = features['label']
    return img,label
