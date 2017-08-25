import h5py
import numpy as np
import cv2
import utils
from random import randint


class Dataset(object):
    def __init__(self, inputs, outputs):
        self._inputs = inputs
        self._outputs = outputs
        self._checkpoint = 0

    def next_batch(self, batch_size):
        i_batch, o_batch = [], []

        j = self._checkpoint
        for i in range(batch_size):
            i_batch.append(self._inputs[j])
            o_batch.append(self._outputs[j])
            j += randint(1, 10)
            if j >= len(self._inputs):
                j -= len(self._inputs)

        self._checkpoint = j
        return i_batch, o_batch


def load_part(filename, start_frame, end_frame, pace):
    log = h5py.File('../data/log/' + filename + '.h5')
    cam = h5py.File('../data/camera/' + filename + '.h5')

    frame_stamp = log['cam1_ptr']
    angles = log['steering_angle']
    speeds = log['speed']
    training_output = []
    training_input = []

    for i in range(start_frame, end_frame, pace):
        output = [utils.encode_angle(angles[i]), utils.encode_speed(speeds[i])]
        training_output.append(np.array(output))
        img = np.transpose(cam['X'][frame_stamp[i]], (1, 2, 0))
        # img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_CUBIC)
        training_input.append(img / 256)

        if i % 1000 == 0:
            print('loaded {}/{}'.format(i - start_frame,
                                        end_frame - start_frame))

    training_data = Dataset(training_input, training_output)

    test_filename = '7'

    log = h5py.File('../data/log/' + test_filename + '.h5')
    cam = h5py.File('../data/camera/' + test_filename + '.h5')

    frame_stamp = log['cam1_ptr']
    angles = log['steering_angle']
    speeds = log['speed']
    test_input = []
    test_output = []

    for i in range(10000, min(len(frame_stamp), 40000), 5):
        output = [min(1.0, angles[i] / 300.0), min(1.0, speeds[i] / 50.0)]
        test_output.append(output)
        img = np.transpose(cam['X'][frame_stamp[i]], (1, 2, 0))
        # img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_CUBIC)
        img = img / 256
        test_input.append(img)

    test_data = Dataset(test_input, test_output)

    return training_data, test_data


def load_data():
    return load_part("1", 20000, 150000, 5)


def load_full_data():
    training_data = None
    test_data = None
    return training_data, test_data


def load_extended_data():
    training_data = None
    test_data = None
    return training_data, test_data


def main():
    load_data()


if __name__ == '__main__':
    main()
