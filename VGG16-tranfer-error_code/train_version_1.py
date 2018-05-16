import numpy as np
import tensorflow as tf
cnn_model_save_path = "cnn-model/cnn_model.ckpt"

files = tf.train.match_filenames_once("train.tfrecord")

filename_queue = tf.train.string_input_producer(files, shuffle=True)

reader = tf.TFRecordReader()

_, serialized_example = reader.read(filename_queue)

features = tf.parse_single_example(
    serialized_example,
    features={
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64),
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'channels': tf.FixedLenFeature([], tf.int64),
    }
)
image, label = features['image'], features['label']
height = tf.cast(features['height'], tf.int32)
width = tf.cast(features['width'], tf.int32)
channels = tf.cast(features['channels'], tf.int32)
# 图片解码
decoded_image = tf.image.decode_jpeg(image)
# 图片转换类型
decoded_image = tf.image.convert_image_dtype(decoded_image, dtype=tf.float32)

image = tf.reshape(decoded_image, [height, width, 3])
image = tf.image.resize_images(image, [224, 224], method=np.random.randint(0, 3))
# image_size = 288
# images = tf.cast(decoded_images, tf.float32)

min_after_dequeue = 1000
batch_size = 10
capacity = min_after_dequeue + 3 * batch_size

image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=capacity,
                                                  min_after_dequeue=min_after_dequeue)

vgg16_npy_path = 'vgg16.npy'
data_dict = np.load(vgg16_npy_path, encoding='latin1').item()


def max_pool(input, name):
    return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def conv_layer(input, name):
    with tf.variable_scope(name):  # CNN's filter is constant, NOT Variable that can be trained
        conv = tf.nn.conv2d(input, data_dict[name][0], [1, 1, 1, 1], padding='SAME')
        lout = tf.nn.relu(tf.nn.bias_add(conv, data_dict[name][1]))
        return lout


def deep_v(x):
    conv1_1 = conv_layer(x, "conv1_1")
    conv1_2 = conv_layer(conv1_1, "conv1_2")
    pool1 = max_pool(conv1_2, 'pool1')

    conv2_1 = conv_layer(pool1, "conv2_1")
    conv2_2 = conv_layer(conv2_1, "conv2_2")
    pool2 = max_pool(conv2_2, 'pool2')

    conv3_1 = conv_layer(pool2, "conv3_1")
    conv3_2 = conv_layer(conv3_1, "conv3_2")
    conv3_3 = conv_layer(conv3_2, "conv3_3")
    pool3 = max_pool(conv3_3, 'pool3')

    conv4_1 = conv_layer(pool3, "conv4_1")
    conv4_2 = conv_layer(conv4_1, "conv4_2")
    conv4_3 = conv_layer(conv4_2, "conv4_3")
    pool4 = max_pool(conv4_3, 'pool4')

    conv5_1 = conv_layer(pool4, "conv5_1")
    conv5_2 = conv_layer(conv5_1, "conv5_2")
    conv5_3 = conv_layer(conv5_2, "conv5_3")
    pool5 = max_pool(conv5_3, 'pool5')

    flatten = tf.reshape(pool5, [-1, 7 * 7 * 512])
    fc6 = tf.layers.dense(flatten, 256, tf.nn.relu, name='fc6')
    out = tf.layers.dense(fc6, 2, name='out')
    print(out)
    return out


out = deep_v(image_batch)

with tf.name_scope('loss'):
    label_batch = tf.cast(label_batch, tf.int64)
    print(label_batch)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_batch,
                                                                   logits=out)
cross_entropy = tf.reduce_mean(cross_entropy)

with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope('test_accuracy'):
    # tf.equal 对比两个矩阵是否相同

    correct_prediction = tf.equal(tf.argmax(out, 1), label_batch)
    # tf.cast 类型转换
    correct_prediction = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(1000):
        if i % 10 == 0:
            print('step %d, train accuracy %g' % (i, accuracy.eval()))
        sess.run(train_step)
        if i == 600:
            print(sess.run(out))
            print(sess.run(tf.argmax(out, 1)))
            print(sess.run(label_batch))
    coord.request_stop()
    coord.join(threads=threads)
