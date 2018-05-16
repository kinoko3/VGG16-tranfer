import tensorflow as tf
import os
import numpy as np
import pandas as pd

file = ['test.tfrecord']


def parse(record):
    features = tf.parse_single_example(
        record,
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
    # 图片解码
    decoded_image = tf.image.decode_jpeg(image)
    # 图片转换类型
    decoded_image = tf.image.convert_image_dtype(decoded_image, dtype=tf.float32)

    image = tf.reshape(decoded_image, [height, width, 3])
    image = tf.image.resize_images(image, [224, 224], method=np.random.randint(0, 3))
    return image, label


dataset = tf.data.TFRecordDataset(file)

datatset = dataset.map(parse)

test_dataset = datatset.batch(batch_size=50)

test_iterator = test_dataset.make_initializable_iterator()
# 读取数据，可用于进一步计算
test_image_batch, test_label_batch = test_iterator.get_next()

with tf.Session() as sess:
    # 1=狗，0=猫
    sess.run(test_iterator.initializer)
    ID = []
    LABEL = []
    with open('model.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        for i in range(1, 101):
            image_batch, label_batch = sess.run([test_image_batch, test_label_batch])
            if i > 50:
                print(i)
                c = tf.import_graph_def(graph_def, input_map={'Placeholder_3:0': image_batch, 'Placeholder_5:0': 1.0},
                                        return_elements=["out/BiasAdd:0"])
                ID.extend(label_batch)

                LABEL.extend(sess.run(tf.argmin(c[0], 1)))
            # print(label_batch.eval())
            # print(image_batch.eval().shape)
            # print(sess.run(tf.argmin(c[0], 1)))
            if i == 100:
                data = {'id': ID, 'label': LABEL}
                frame = pd.DataFrame(data)
                frame.to_csv('output_1000000.csv', index=False, columns=['id', 'label'])

#             if i % 10 == 0 and i != 0:
#                 data = {'id': ID, 'label': LABEL}
#                 frame = pd.DataFrame(data)
#                 frame.to_csv('output'+'_'+str(i)+'.csv', index=False, columns=['id', 'label'])
#                 ID = []
#                 LABEL = []
