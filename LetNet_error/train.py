import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np

# IMG_HEIGHT = 277
# IMG_WIDTH = 277
# IMG_CHANNELS = 3
# NUM_TRAIN = 1000
#
# filenames = os.listdir('train')
files = tf.train.match_filenames_once("tfrecord/train.tfrecord")

filename_queue = tf.train.string_input_producer(files, shuffle=False)

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
image_data = tf.image.rgb_to_grayscale(image)
image = tf.image.resize_images(image_data, [288, 288],  method=np.random.randint(0, 3))


# image_size = 288
# images = tf.cast(decoded_images, tf.float32)

min_after_dequeue = 1000
batch_size = 10
capacity = min_after_dequeue + 3 * batch_size

image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=capacity,
                                                                min_after_dequeue=min_after_dequeue)


# name_scope命名空间
def deepnn(x):

    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 288, 288, 1])

    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_varibale([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1)+ b_conv1)

    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_varibale([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('conv3'):
        W_conv3 = weight_variable([5, 5, 64, 128])
        b_conv3 = bias_varibale([128])
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    with tf.name_scope('pool3'):
        h_pool3 = max_pool_2x2(h_conv3)

    with tf.name_scope('conv4'):
        W_conv4 = weight_variable([5, 5, 128, 256])
        b_conv4 = bias_varibale([256])
        h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
    with tf.name_scope('pool4'):
        h_pool4 = max_pool_2x2(h_conv4)

    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([18 * 18 * 256, 4096])
        b_fc1 = bias_varibale([4096])

        h_pool4_flat = tf.reshape(h_pool4, [-1, 18 * 18 * 256])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # Map the 1024 features to 10 classes, one for each digit
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([4096, 2])
        b_fc2 = bias_varibale([2])

        y_conv = tf.add(tf.matmul(h_fc1_drop, W_fc2), b_fc2)
        print(y_conv)

    return y_conv, keep_prob

# 参数中步长strides=[1, 1, 1, 1]第一个和最后一个必须要求使用1, 卷积层的步长只对矩阵长和宽有效
# W卷积层权重，共享权重
# 快速前向传播
# conv2d过滤器尺寸，


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# ksize过滤器尺寸，
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 初始化权重
def weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial)


# 初始化偏置
def bias_varibale(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# Build the graph for the deep net
y_conv, keep_prob = deepnn(image_batch)

with tf.name_scope('loss'):
    label_batch = tf.cast(label_batch, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_batch,
                                                        logits=y_conv)


# 获得均值交叉验证
cross_entropy = tf.reduce_mean(cross_entropy)

# 获得优化器(梯度下降，反向传播)
with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 测试集，使用训练好模型进行测试
with tf.name_scope('accuracy'):
    # tf.equal 对比两个矩阵是否相同

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), label_batch)
    # tf.cast 类型转换
    correct_prediction = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

tf.summary.scalar("loss", cross_entropy)
merged_summary = tf.summary.merge_all()
train_writer = tf.summary.FileWriter("model")
train_writer.add_graph(tf.get_default_graph())
# saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    run_metadata = tf.RunMetadata()

    for i in range(100):
        if i % 10 == 0:
            train_accuracy = accuracy.eval(feed_dict={keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
            print('训练10轮')
        _, summary = sess.run([train_step, merged_summary], feed_dict={keep_prob: 0.5}, run_metadata=run_metadata, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE))
        train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
        train_writer.add_summary(summary, i)

    train_writer.close()
    coord.request_stop()
    coord.join(threads=threads)

