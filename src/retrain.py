import tensorflow as tf
from time import time

import data_loader

training_data, testing_data = data_loader.load_data(checkpoint=5*50*400)

session = tf.InteractiveSession()
saver = tf.train.import_meta_graph('aya399-399.meta')
saver.restore(session, tf.train.latest_checkpoint('./'))

graph = tf.get_default_graph()
inputs_ = graph.get_tensor_by_name('inputs')
targets_ = graph.get_tensor_by_name('targets')

loss = graph.get_tensor_by_name('loss')
opt = graph.get_tensor_by_name('opt')

for it in range(10000):
    start_time = time()
    batch_xs, batch_ys = training_data.next_batch(50)
    session.run(opt, feed_dict={inputs_: batch_xs, targets_: batch_ys})
    print("Iteration: {}".format(it))

    if it % 100 == 99:
        batch_xs, batch_ys = testing_data.next_batch(100)
        error = session.run(loss, feed_dict={inputs_: batch_xs, targets_: batch_ys})
        print("Iteration: {}, Loss: {}, execution time: {}".format(it, error, time() - start_time))
        saver.save(session, 'aya{}'.format(it), global_step=it)

saver.save(session, 'aya', global_step=10000)
