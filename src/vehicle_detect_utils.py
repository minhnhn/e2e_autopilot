import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from skimage.feature import hog
from keras.models import model_from_json
import pickle


def load_data():
    vehicle_data = '/Users/robert/dev/personal/e2e_autopilot/data/vehicle_detection/vehicles/*'
    non_vehicle_data = '/Users/robert/dev/personal/e2e_autopilot/data/vehicle_detection/non-vehicles/*'
    vehicle_data_dirs = glob(vehicle_data)
    non_vehicle_data_dirs = glob(non_vehicle_data)

    v_img_names = np.concatenate(np.array([glob(v_dir + '/*.png') for v_dir in vehicle_data_dirs]))
    nv_img_names = np.concatenate(np.array([glob(nv_dir + '/*.png') for nv_dir in non_vehicle_data_dirs]))

    v_images = []
    for img_name in tqdm(v_img_names):
        img = cv2.imread(img_name)
        # features = [hog(img[:, :, i], block_norm='L2-Hys').flatten() for i in range(3)]
        # features = np.concatenate(np.array(features))
        img = cv2.resize(img, (16, 16))
        v_images.append(img)

    nv_images = []
    for img_name in tqdm(nv_img_names):
        img = cv2.imread(img_name)
        # features = [hog(img[:, :, i], block_norm='L2-Hys').flatten() for i in range(3)]
        # features = np.concatenate(np.array(features))
        img = cv2.resize(img, (16, 16))
        nv_images.append(img)

    v_images = np.array(v_images)
    nv_images = np.array(nv_images)

    nv_labels = np.zeros(nv_images.shape[0])
    v_labels = np.ones(v_images.shape[0])

    inputs = np.concatenate([v_images, nv_images])
    labels = np.concatenate([v_labels, nv_labels])

    return inputs, labels


def get_heat_map(img, model):
    height, width, _ = img.shape
    heat_map = np.zeros([height, width])

    horizon = 30
    i = height
    windows = []
    features = []

    while i > horizon:
        window = int(i * 0.75)
        for j in range(0, width, int(window / 4)):
            if (j + window) >= width:
                break

            tmp = img[i - window:i, j:j + window]

            # using HOG
            tmp = cv2.resize(tmp, (64, 64))
            feature = [hog(tmp[:, :, i], block_norm='L2-Hys').flatten() for i in range(3)]
            feature = np.concatenate(np.array(feature))
            windows.append([i - window, i, j, j + window])
            features.append(feature)
            # heat_map[i - window:i, j:j + window] += model.predict(np.array([features]))[0]

            # using image
            # tmp = cv2.resize(tmp, (16, 16))
            # heat_map[i - window:i, j:j + window] += model.predict(np.array([tmp]))
        i = max(int(i * 0.8), i - 20)

    preds = model.predict(np.array(features))

    for window, feature, value in zip(windows, features, preds):
        heat_map[window[0]:window[1], window[2]:window[3]] += value

    return heat_map


def get_clf_model(model_name='random_forest'):
    if model_name == 'random_forest':
        model = pickle.load(open('rf_model_128_1.pkl', 'rb'))
    else:
        json_file = open('/Users/robert/dev/personal/e2e_autopilot/src/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights("/Users/robert/dev/personal/e2e_autopilot/src/model.h5")

    return model


# inputs, labels = load_data()
# print(inputs.shape)
# pickle.dump(inputs, open('vehicle_image_inputs.pkl', 'wb'))
# pickle.dump(labels, open('vehicle_labels.pkl', 'wb'))

