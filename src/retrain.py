import tensorflow as tf
from time import time

import data_loader

MODEL_PATH = "../models/"

try:
    with open(MODEL_PATH+'data_checkpoint.txt', 'r') as f:
        DATA_CHECKPOINT = int(f.readline().split()[0])
except Exception:
    DATA_CHECKPOINT = 0

training_data, testing_data = data_loader.load_data(checkpoint=DATA_CHECKPOINT)

session = tf.InteractiveSession()
saver = tf.train.import_meta_graph(MODEL_PATH + 'aya-5350.meta')
saver.restore(session, tf.train.latest_checkpoint(MODEL_PATH + './'))

graph = tf.get_default_graph()
inputs_ = graph.get_tensor_by_name('inputs:0')
targets_ = graph.get_tensor_by_name('targets:0')

loss = graph.get_tensor_by_name('loss:0')
opt = graph.get_operation_by_name('opt')

for it in range(5351, 10000):
    start_time = time()
    batch_xs, batch_ys = training_data.next_batch(100)

    data_checkpoint = training_data.checkpoint

    session.run(opt, feed_dict={inputs_: batch_xs, targets_: batch_ys})
    print("Iteration: {} , Execution time: {}".format(it, time() - start_time))

    if it % 20 == 0:
        batch_xs, batch_ys = testing_data.next_batch(200)
        error = session.run(loss, feed_dict={inputs_: batch_xs, targets_: batch_ys})
        print("Iteration: {}, Loss: {}, execution time: {}".format(it, error, time() - start_time))
        saver.save(session, MODEL_PATH + 'aya', global_step=it)
        with open(MODEL_PATH + "data_checkpoint.txt", 'w') as f:
            f.write(str(data_checkpoint))

saver.save(session, MODEL_PATH + 'aya', global_step=10000)
