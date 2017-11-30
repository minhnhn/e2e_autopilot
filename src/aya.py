import tensorflow as tf
from matplotlib import pylab as pl
import h5py
import cv2
import numpy as np
import utils
import matplotlib.pyplot as plt
import utils
from time import time
import vehicle_detect_utils

MODEL_PATH = '../models/'
HEAT_MAP_DECAY_FACTOR = 0.6

session = tf.InteractiveSession()
saver = tf.train.import_meta_graph(MODEL_PATH + 'aya-5440.meta')
saver.restore(session, tf.train.latest_checkpoint(MODEL_PATH + './'))

graph = tf.get_default_graph()
inputs_ = graph.get_tensor_by_name('inputs:0')
targets_ = graph.get_tensor_by_name('targets:0')

loss = graph.get_tensor_by_name('loss:0')
outputs = graph.get_tensor_by_name('outputs:0')
log = h5py.File("../data/log/2.h5")
cam = h5py.File("../data/camera/2.h5")

frame_stamp = log['cam1_ptr']

img = cam['X'][frame_stamp[75100]]
img = np.transpose(img, (1, 2, 0))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# cv2.imwrite('../data/vehicle.jpg', img)

pl.ion()
img = utils.lane_extraction(img)
my_img = pl.imshow(img)

current_angle = 0
delta = 0
heat_map = None
detection_model = vehicle_detect_utils.get_clf_model()

for i in range(75000, 200000, 20):
    crt_time = time()
    img = cam['X'][frame_stamp[i]]
    angle = log['steering_angle'][i]
    speed = log['speed'][i]

    img = np.transpose(img, (1, 2, 0))

    if heat_map is None:
        heat_map = vehicle_detect_utils.get_heat_map(img, detection_model)
    else:
        heat_map = heat_map * HEAT_MAP_DECAY_FACTOR + \
                   (1 - HEAT_MAP_DECAY_FACTOR) * vehicle_detect_utils.get_heat_map(img, detection_model)

    inp = [img / 256]
    res = session.run(outputs, feed_dict={inputs_: inp})
    pred_angle = utils.decode_angle(res[0][0])
    pred_speed = utils.decode_speed(res[0][1])

    current_angle = utils.angle_shift(current_angle, pred_angle)

    tmp = utils.lane_extraction(img)
    if tmp is not None:
        img = tmp

    uint8_heat_map = cv2.convertScaleAbs(heat_map)

    _, uint8_heat_map = cv2.threshold(uint8_heat_map, .5, 255, cv2.THRESH_BINARY)

    heat_map_edges = cv2.Canny(uint8_heat_map, .5, 3)

    img[:, :, 0] = cv2.addWeighted(img[:, :, 0], 1, np.uint8(heat_map_edges), 1, 0)

    utils.draw_path_on(img, speed, -angle / 10.0, color=(255, 0, 0))
    utils.draw_path_on(img, pred_speed, -current_angle / 10.0, color=(0, 255, 0))
    print(time() - crt_time)

    # log details

    if abs(angle) < 30:
        print('{}    Straight    {:.4f}    speed = {:.4f}'.format(i, angle, speed))
    elif angle < 0:
        print('{}    Left        {:.4f}    speed = {:.4f}'.format(i, angle, speed))
    else:
        print('{}    Right       {:.4f}    speed = {:.4f}'.format(i, angle, speed))

    # display
    # if img is None:
    #     print(i)
    #     continue

    my_img.set_data(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # cv2.imwrite('{}.jpg'.format(i), img)
    pl.pause(.0001)
    pl.draw()


