import tensorflow as tf
from matplotlib import pylab as pl
import h5py
import cv2
import numpy as np
import utils

MODEL_PATH = '../models/'

session = tf.InteractiveSession()
saver = tf.train.import_meta_graph(MODEL_PATH + 'aya-5350.meta')
saver.restore(session, tf.train.latest_checkpoint(MODEL_PATH + './'))

graph = tf.get_default_graph()
inputs_ = graph.get_tensor_by_name('inputs:0')
targets_ = graph.get_tensor_by_name('targets:0')

loss = graph.get_tensor_by_name('loss:0')
outputs = graph.get_tensor_by_name('outputs:0')

log = h5py.File("../data/log/2.h5")
cam = h5py.File("../data/camera/2.h5")

frame_stamp = log['cam1_ptr']

img = cam['X'][frame_stamp[70100]]
img = np.transpose(img, (1, 2, 0))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
pl.ion()
my_img = pl.imshow(img)

for i in range(60000, 200000, 40):
    img = cam['X'][frame_stamp[i]]
    angle = log['steering_angle'][i]
    speed = log['speed'][i]

    img = np.transpose(img, (1, 2, 0))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    inp = [img / 256]
    res = session.run(outputs, feed_dict={inputs_: inp})
    # print(res)
    # print(res.shape)
    pred_angle = utils.decode_angle(res[0][0])
    pred_speed = utils.decode_speed(res[0][1])
    utils.draw_path_on(img, speed, -angle / 10.0, color=(255, 0, 0))
    utils.draw_path_on(img, pred_speed, -pred_angle / 10.0, color=(0, 255, 0))
    # cv2.imwrite("../temp/img{}.png".format(i), img)
    # print(img.shape)
    # img = cv2.flip(img, 180)
    # out_vid.write(img)
    # print(img.shape)

    # resize image
    # img = cv2.resize(img,(2*img.shape[1], 2*img.shape[0]), interpolation = cv2.INTER_CUBIC)
    # edges = cv2.Canny(cv2.cvtColor(img , cv2.COLOR_RGB2GRAY) ,50,150)
    # edges = cv2.GaussianBlur(edges , (3,3) , 0)
    # res = np.hstack((cv2.cvtColor(img , cv2.COLOR_RGB2GRAY) , edges))

    # log details

    if abs(angle) < 30:
        print('{}    Straight    {:.4f}    speed = {:.4f}'.format(i, angle, speed))
    elif angle < 0:
        print('{}    Left        {:.4f}    speed = {:.4f}'.format(i, angle, speed))
    else:
        print('{}    Right       {:.4f}    speed = {:.4f}'.format(i, angle, speed))

    # display

    my_img.set_data(img)
    pl.pause(.0001)
    pl.draw()


