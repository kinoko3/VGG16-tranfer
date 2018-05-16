import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np

IMG_HEIGHT = 277
IMG_WIDTH = 277
IMG_CHANNELS = 3
NUM_TRAIN = 1000

filenames = os.listdir('train/train')

train_file = 'train.tfrecord'
test_file = 'test.tfrecord'


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


with tf.Session() as sess:
    with tf.python_io.TFRecordWriter(train_file) as writer:
        for i, name in enumerate(filenames):
            if name[:-9] == 'dog':
                label = np.argmax(np.array([1, 0]))
            else:
                label = np.argmax(np.array([0, 1]))
            image_raw_data = tf.gfile.FastGFile('train/train/'+name, 'rb').read()
            img_data = tf.image.decode_jpeg(image_raw_data)
            img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
            # image_data = tf.image.rgb_to_grayscale(img_data)
            # image_data = tf.image.resize_images(image_data, [288, 288], method=np.random.randint(0, 3))
            img_raw = img_data.eval()
            height, width, channels = img_raw.shape
            # image_raw = img_raw.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'image': _bytes_feature(image_raw_data),
                'label': _int64_feature(label),
                'height': _int64_feature(height),
                'width': _int64_feature(width),
                'channels': _int64_feature(channels)
            }))
            writer.write(example.SerializeToString())
            print('训练集整合 '+str(i))
            if i == 13000:
                break

    with tf.python_io.TFRecordWriter(test_file) as writer:
        for i, name in enumerate(filenames):
            if i < 13000:
                continue
            if name[:-9] == 'cat':
                label = np.argmax(np.array([1, 0]))
            else:
                label = np.argmax(np.array([0, 1]))
            image_raw_data = tf.gfile.FastGFile('train/train/' + name, 'rb').read()
            img_data = tf.image.decode_jpeg(image_raw_data)
            img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
            # image_data = tf.image.rgb_to_grayscale(img_data)
            # image_data = tf.image.resize_images(image_data, [288, 288], method=np.random.randint(0, 3))
            img_raw = img_data.eval()
            height, width, channels = img_raw.shape
            # image_raw = img_raw.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'image': _bytes_feature(image_raw_data),
                'label': _int64_feature(label),
                'height': _int64_feature(height),
                'width': _int64_feature(width),
                'channels': _int64_feature(channels)
            }))
            writer.write(example.SerializeToString())
            print('测试集整合 ' + str(i))
            if i == 17000:
                break
