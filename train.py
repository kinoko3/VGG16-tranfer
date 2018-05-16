import numpy as np
import tensorflow as tf

cnn_model_save_path = "cnn-model/cnn_model.ckpt"



def distort_color(image, color_ordering=0):
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    else:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)

    return tf.clip_by_value(image, 0.0, 1.0)


def data_read(filename):
    files = tf.train.match_filenames_once(filename)

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
    image = tf.image.random_flip_left_right(image)
    image = distort_color(image, np.random.randint(2))
    # image_size = 288
    # images = tf.cast(decoded_images, tf.float32)

    min_after_dequeue = 1000
    batch_size = 100
    capacity = min_after_dequeue + 3 * batch_size

    image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=capacity,
                                                      min_after_dequeue=min_after_dequeue)
    return image_batch, label_batch


vgg16_npy_path = 'vgg16.npy'
data_dict = np.load(vgg16_npy_path, encoding='latin1').item()


def max_pool(input, name):
    return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def conv_layer(input, name):
    with tf.variable_scope(name):  # CNN's filter is constant, NOT Variable that can be trained
        conv = tf.nn.conv2d(input, data_dict[name][0], [1, 1, 1, 1], padding='SAME')
        lout = tf.nn.relu(tf.nn.bias_add(conv, data_dict[name][1]))
        return lout


def conv_layer_train(input, name):
    with tf.variable_scope(name):  # CNN's filter is constant, Have Variable that can be trained
        conv = tf.nn.conv2d(input, tf.Variable(data_dict[name][0]), [1, 1, 1, 1], padding='SAME', )
        lout = tf.nn.relu(tf.nn.bias_add(conv, tf.Variable(data_dict[name][1])))
        return lout


# 初始化权重
def weight_variable(name):
    return tf.Variable(data_dict[name][0])


# 初始化偏置
def bias_varibale(name):
    return tf.Variable(data_dict[name][1])


def deep_v(x, keep_prob):
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

    conv5_1 = conv_layer_train(pool4, "conv5_1")
    conv5_2 = conv_layer_train(conv5_1, "conv5_2")
    conv5_3 = conv_layer_train(conv5_2, "conv5_3")
    pool5 = max_pool(conv5_3, 'pool5')

    flatten = tf.reshape(pool5, [-1, 7 * 7 * 512])

    fc6 = tf.layers.dense(flatten, 256, tf.nn.relu, name='fc6',
                          kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    fc6_drop = tf.nn.dropout(fc6, keep_prob=keep_prob)
    out = tf.layers.dense(fc6_drop, 2, name='out', kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    print(out)
    return out


train_image_batch, train_label_batch = data_read('train.tfrecord')

train_label = tf.one_hot(train_label_batch, 2, 1, 0)

test_image_batch, test_label_batch = data_read('test.tfrecord')

test_label = tf.one_hot(test_label_batch, 2, 1, 0)

x = tf.placeholder(tf.float32, [None, 224, 224, 3])
print(x)
y = tf.placeholder(tf.float32, [None, 2])  # label
print(y)
keep_prob = tf.placeholder(tf.float32)
print(keep_prob)

out = deep_v(x, keep_prob)

with tf.name_scope('loss'):
    label_batch = tf.cast(y, tf.int64)
    # print(label_batch)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label_batch,
                                                               logits=out)
cross_entropy = tf.reduce_mean(cross_entropy)

with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    # tf.equal 对比两个矩阵是否相同

    correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
    # tf.cast 类型转换
    correct_prediction = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

tf.summary.scalar('accuracy', accuracy)
tf.summary.scalar("loss", cross_entropy)
merged_summary = tf.summary.merge_all()
train_writer = tf.summary.FileWriter("model")
train_writer.add_graph(tf.get_default_graph())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    train_image_batch, train_label = sess.run([train_image_batch, train_label])
    test_image_batch, test_label = sess.run([test_image_batch, test_label])
    run_metadata = tf.RunMetadata()
    max_acc = 0
    saver = tf.train.Saver()  # 模型保存
    for i in range(1000):
        if i % 10 == 0:
            acc = accuracy.eval(feed_dict={x: test_image_batch, y: test_label, keep_prob: 1.0})
            print('step %d,train accuracy %g' %
                  (i, acc))
            if max_acc < acc:
                max_acc = acc
                saver.save(sess, save_path=cnn_model_save_path)
        summary, _ = sess.run([merged_summary, train_step],
                              feed_dict={x: train_image_batch, y: train_label, keep_prob: 0.5})

        train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
        train_writer.add_summary(summary, i)
    train_writer.close()
    coord.request_stop()
    coord.join(threads=threads)