import tensorflow as tf
from time import time
import os

import data_loader

MODEL_PATH = "../models/"
DATA_CHECKPOINT = 0


def train(training_data, testing_data):
    inputs_ = tf.placeholder(tf.float32, [None, 160, 320, 3], name='inputs')

    targets_ = tf.placeholder(tf.float32, [None, 2], name='targets')

    # now size 160*320*3
    conv1 = tf.layers.conv2d(inputs_, 24, (5, 5), padding='same', activation=tf.nn.relu)
    maxpool1 = tf.layers.max_pooling2d(conv1, (2, 2), (2, 2), padding='same')

    # now size 80*160*24
    conv2 = tf.layers.conv2d(maxpool1, 36, (5, 5), padding='same', activation=tf.nn.relu)
    maxpool2 = tf.layers.max_pooling2d(conv2, (2, 2), (2, 2), padding='same')

    # now size 40*80*36
    conv3 = tf.layers.conv2d(maxpool2, 48, (5, 5), padding='same', activation=tf.nn.relu)
    maxpool3 = tf.layers.max_pooling2d(conv3, (2, 2), (2, 2), padding='same')

    # now size 20*40*48
    conv4 = tf.layers.conv2d(maxpool3, 48, (3, 3), padding='same', activation=tf.nn.relu)
    maxpool4 = tf.layers.max_pooling2d(conv4, (2, 2), (2, 2), padding='same')

    # now size 10*20*48
    conv5 = tf.layers.conv2d(maxpool4, 64, (3, 3), padding='same', activation=tf.nn.relu)
    maxpool5 = tf.layers.max_pooling2d(conv5, (2, 2), (2, 2), padding='same')

    # now size 5*10*64
    # tflat = tf.reshape(maxpool5, (None, 5 * 10 * 64))
    tflat = tf.reshape(maxpool5, [-1, 5 * 10 * 64])

    # now size 3200
    fc1 = tf.layers.dense(tflat, 100, tf.nn.relu)

    # now size 100
    fc2 = tf.layers.dense(fc1, 50, tf.nn.relu)

    # output size
    logits = tf.layers.dense(fc2, 2, activation=None)

    outputs = tf.nn.sigmoid(logits, name='outputs')
    loss = tf.reduce_mean(tf.squared_difference(outputs, targets_), name='loss')

    opt = tf.train.AdamOptimizer(1e-4, name='opt').minimize(loss)

    saver = tf.train.Saver()
    session = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    for it in range(1000):
        start_time = time()
        batch_xs, batch_ys = training_data.next_batch(50)

        data_checkpoint = training_data.checkpoint
        with open(MODEL_PATH + "data_checkpoint.txt", 'w') as f:
            f.write(str(data_checkpoint))

        session.run(opt, feed_dict={inputs_: batch_xs, targets_: batch_ys})
        print("Iteration: {}, execution time: {}".format(it, time() - start_time))
        
        if it % 100 == 99:
            batch_xs, batch_ys = testing_data.next_batch(100)
            error = session.run(loss, feed_dict={inputs_: batch_xs, targets_: batch_ys})
            print("Iteration: {}, Loss: {}, execution time: {}".format(it, error, time() - start_time))
            saver.save(session, MODEL_PATH + 'aya{}'.format(it), global_step=it)

    saver.save(session, MODEL_PATH + 'aya', global_step=1000)

if __name__ == '__main__':
    training_data, testing_data = data_loader.load_data(checkpoint=DATA_CHECKPOINT)
    train(training_data, testing_data)


