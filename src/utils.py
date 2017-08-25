import cv2
import numpy as np
from skimage import transform as tf

rsrc = \
    [[43.45456230828867, 118.00743250075844],
     [104.5055617352614, 69.46865203761757],
     [114.86050156739812, 60.83953551083698],
     [129.74572757609468, 50.48459567870026],
     [132.98164627363735, 46.38576532847949],
     [301.0336906326895, 98.16046448916306],
     [238.25686790036065, 62.56535881619311],
     [227.2547443287154, 56.30924933427718],
     [209.13359962247614, 46.817221154818526],
     [203.9561297064078, 43.5813024572758]]
rdst = \
    [[10.822125594094452, 1.42189132706374],
     [21.177065426231174, 1.5297552836484982],
     [25.275895776451954, 1.42189132706374],
     [36.062291434927694, 1.6376192402332563],
     [40.376849698318004, 1.42189132706374],
     [11.900765159942026, -2.1376192402332563],
     [22.25570499207874, -2.1376192402332563],
     [26.785991168638553, -2.029755283648498],
     [37.033067044190524, -2.029755283648498],
     [41.67121717733509, -2.029755283648498]]

max_angle_shift_rate = 0.05

tform3_img = tf.ProjectiveTransform()
tform3_img.estimate(np.array(rdst), np.array(rsrc))


def perspective_tform(x, y):
    p1, p2 = tform3_img((x, y))[0]
    return p2, p1


def draw_without_transform(img, x, y, color, sz=1):
    row, col = x, y
    if 0 <= row < img.shape[0] and 0 <= col < img.shape[1]:
        img[int(row)-sz:int(row)+sz, int(col)-sz:int(col)+sz] = color


def draw_pt(img, x, y, color, sz=1):
    row, col = perspective_tform(x, y)
    if 0 <= row < img.shape[0] and 0 <= col < img.shape[1]:
        img[int(row)-sz:int(row)+sz, int(col)-sz:int(col)+sz] = color


def draw_path(img, path_x, path_y, color):
    for x, y in zip(path_x, path_y):
        draw_pt(img, x, y, color)


def calc_curvature(v_ego, angle_steers, angle_offset=0):
    deg_to_rad = np.pi / 180.
    slip_fator = 0.0014  # slip factor obtained from real data
    steer_ratio = 15.3  # from http://www.edmunds.com/acura/ilx/2016/road-test-specs/
    wheel_base = 2.67  # from http://www.edmunds.com/acura/ilx/2016/sedan/features-specs/

    angle_steers_rad = (angle_steers - angle_offset) * deg_to_rad
    curvature = angle_steers_rad / (steer_ratio * wheel_base * (1. + slip_fator * v_ego ** 2))
    return curvature


def calc_lookahead_offset(v_ego, angle_steers, d_lookahead, angle_offset=0):
    # *** this function returns the lateral offset given the steering angle, speed and the lookahead distance
    curvature = calc_curvature(v_ego, angle_steers, angle_offset)

    # clip is to avoid arcsin NaNs due to too sharp turns
    y_actual = d_lookahead * np.tan(np.arcsin(np.clip(d_lookahead * curvature, -0.999, 0.999)) / 2.)
    return y_actual, curvature


def draw_path_on(img, speed_ms, angle_steers, color=(0, 0, 255)):
    path_x = np.arange(0., 50.1, 0.5)
    path_y, _ = calc_lookahead_offset(speed_ms, angle_steers, path_x)
    draw_path(img, path_x, path_y, color)


def get_curvature(img, y0, angle_steers, speed_ms=25):
    path_x = np.arange(0., 40.1, 0.5)
    path_y, _ = calc_lookahead_offset(speed_ms, angle_steers, path_x)

    res = []
    prev = img.shape[0]

    for (x, y) in zip(path_x, path_y):
        tx, ty = perspective_tform(x, y)
        tx = tx * img.shape[0] / 160

        # for FPT video frame
        # tx = tx / 2 + (img.shape[0] / 2)
        ty = ty * img.shape[1] / 320

        ratio_to_vanishing_point = (tx - 16) / (float)(144)
        ty = ty + int(ratio_to_vanishing_point * (y0) * img.shape[1])
        if (tx >= 0) and (tx < img.shape[0]) and (ty >= 0) and (ty < img.shape[1]) and (prev - tx >= 2):
            res.append((tx, ty))
            prev = tx
    return res


def fast_get_curvature(img, y0, angle_steers, density):
    path_x = np.arange(img.shape[0], img.shape[0] * 1 / 4, -density)
    x0 = img.shape[0] * 1 / 6
    delta_x = (float)(img.shape[0]) - x0
    path_y = []
    res = []
    for x in path_x:
        dy = int(1 / ((x - x0) / (float)(delta_x)) * angle_steers * 2.0)
        path_y.append((img.shape[1] / 2) - dy)

    for (x, y) in zip(path_x, path_y):
        tx, ty = x, y
        ratio_to_vanishing_point = (tx - 16) / (float)(144)
        ty = ty + int(ratio_to_vanishing_point * (y0) * img.shape[1])
        res.append((tx, ty))

    return res


def draw_lane_on(img, y0, angle_steers, color=(0, 0, 255)):
    points = get_curvature(img, y0, angle_steers, 6)
    for pt in points:
        draw_without_transform(img, pt[0], pt[1], color)


def encode_angle(angle):
    x = (angle / 500.0) + 0.5
    x = min(1.0, x)
    x = max(-1.0, x)
    return x


def decode_angle(x):
    x -= 0.5
    return x * 500


def encode_speed(speed):
    x = (speed / 50.0) + 0.5
    x = min(1.0, x)
    x = max(-1.0, x)
    return x


def decode_speed(x):
    x -= 0.5
    return x * 50


#	shift angle x to y with rate angle_dv
def angle_shift(x, y):
    return (x * (1 - max_angle_shift_rate) + y * max_angle_shift_rate)
