import cv2
import numpy as np
from skimage import transform as tf
import matplotlib.pyplot as plt
from scipy import signal
from src import config

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

max_angle_shift_rate = 0.2

tform3_img = tf.ProjectiveTransform()
tform3_img.estimate(np.array(rdst), np.array(rsrc))


def perspective_tform(x, y):
    p1, p2 = tform3_img((x, y))[0]
    return p2, p1


def draw_without_transform(img, x, y, color, sz=1):
    row, col = x, y
    if 0 <= row < img.shape[0] and 0 <= col < img.shape[1]:
        img[int(row) - sz:int(row) + sz, int(col) - sz:int(col) + sz] = color


def draw_pt(img, x, y, color, sz=1):
    row, col = perspective_tform(x, y)
    if 0 <= row < img.shape[0] and 0 <= col < img.shape[1]:
        img[int(row) - sz:int(row) + sz, int(col) - sz:int(col) + sz] = color


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


# shift angle x to y with rate angle_dv
def angle_shift(x, y):
    return x * (1 - max_angle_shift_rate) + y * max_angle_shift_rate


def bird_eye_transform(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # darken gray image
    # gray_img = np.array(gray_img, dtype=np.float32)
    # gray_img *= 0.75
    # gray_img = np.uint8(gray_img)

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_img = cv2.blur(hsv_img, (2, 2))

    avg_brightness = gray_img.mean()

    yellow_range = ([20, 30, avg_brightness], [30, 255, 255])
    upper, lower = yellow_range
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    yellow_mask = cv2.inRange(hsv_img, upper, lower)
    yellow_channel = cv2.bitwise_and(hsv_img, hsv_img, mask=yellow_mask)

    # TODO: solve the contrast issue when go from dark to bright
    threshold = int(gray_img.max() * 3 / 4)
    # threshold = 255
    white_range = ([0, 0, threshold], [180, 40, 255])
    upper, lower = white_range
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    white_mask = cv2.inRange(hsv_img, upper, lower)
    white_channel = cv2.bitwise_and(hsv_img, hsv_img, mask=white_mask)

    result = cv2.bitwise_or(white_channel, yellow_channel)
    result = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    _, result = cv2.threshold(result, 50, 255, cv2.THRESH_BINARY)

    result = cv2.bitwise_or(result, gray_img)

    src_points = np.array([[0, 160], [320, 160], [114, 43], [172, 39]], np.float32)
    dst_points = np.array([[200, 160], [400, 160], [140, 30], [360, 30]], np.float32)

    lambda_mat = cv2.getPerspectiveTransform(src_points, dst_points)

    bird_eye = cv2.warpPerspective(result, lambda_mat, (640, 160))
    # edges = cv2.Canny(result, 50, 150)

    # return np.hstack([img, white_channel])
    # return np.hstack([gray_img, result])
    return bird_eye


def add_lane_pixels(img, upper_y, lower_y, mid_x, lane_x, lane_y, window_size=20):
    if mid_x < window_size // 2:
        return

    window = img[lower_y:upper_y, mid_x - window_size // 2:mid_x + window_size // 2]

    if window.mean() > window.max() * 3 / 4:
        return

    x, y = (window.T > window.mean()).nonzero()
    x += mid_x - window_size // 2
    y += lower_y

    lane_x.append(x)
    lane_y.append(y)


def get_second_order_polynomial(x, y):
    poly = np.polyfit(x, y, 2)
    return poly


def fit_polynomial(x, poly):
    return poly[0] * x ** 2 + poly[1] * x + poly[2]


def get_lane_pixels(img, steps=12, medfil_kernel=3, window_size=30):
    img_height, img_width = img.shape
    stripe_height = img_height // steps

    split_pixel = 310

    # left_half = img[:, :split_pixel]
    # right_half = img[:, split_pixel:]
    left_lane_x = []
    left_lane_y = []
    right_lane_x = []
    right_lane_y = []

    # f, axes = plt.subplots(steps, 1)
    for step in range(steps // 2, steps):
        stripe = img[stripe_height * step:stripe_height * (step + 1)]
        histogram = np.sum(stripe, axis=0)
        histogram = signal.medfilt(histogram, kernel_size=medfil_kernel)

        left_peaks = np.array(signal.find_peaks_cwt(histogram[:split_pixel], np.arange(1, 10)))
        # TODO: Fix hard coded right lane bound
        right_peaks = np.array(signal.find_peaks_cwt(histogram[split_pixel:500], np.arange(1, 10)))

        # axes[step].plot(histogram)

        if len(left_peaks) > 0:
            left_peak_hist = list(histogram[left_peaks])
            left_peak = left_peaks[left_peak_hist.index(max(left_peak_hist))]
            # print(left_peak)
            add_lane_pixels(img, stripe_height * (step + 1), stripe_height * step, left_peak, left_lane_x,
                            left_lane_y)

        if len(right_peaks) > 0:
            right_peaks += split_pixel
            right_peak_hist = list(histogram[right_peaks])
            right_peak = right_peaks[right_peak_hist.index(max(right_peak_hist))]

            # print(right_peak)
            add_lane_pixels(img, stripe_height * (step + 1), stripe_height * step, right_peak, right_lane_x,
                            right_lane_y)
            # axes[step].imshow(stripe, cmap='gray')

    if len(left_lane_x) > 0:
        left_lane_x = np.concatenate(left_lane_x)
    if len(left_lane_y) > 0:
        left_lane_y = np.concatenate(left_lane_y)
    if len(right_lane_x) > 0:
        right_lane_x = np.concatenate(right_lane_x)
    if len(right_lane_y) > 0:
        right_lane_y = np.concatenate(right_lane_y)

    return left_lane_x, left_lane_y, right_lane_x, right_lane_y


def get_lane_polynomials(img):
    left_lane_x, left_lane_y, right_lane_x, right_lane_y = get_lane_pixels(img)

    if (len(left_lane_x) == 0) or (len(left_lane_y) == 0) \
            or (len(right_lane_x) == 0) or (len(right_lane_y) == 0):
        return None, None

    left_poly = get_second_order_polynomial(left_lane_y, left_lane_x)
    right_poly = get_second_order_polynomial(right_lane_y, right_lane_x)
    # ys = np.arange(0, img_height)
    # plt.plot(fit_polynomial(ys, left_poly), ys, color='green')
    # plt.plot(fit_polynomial(ys, right_poly), ys, color='green')

    return left_poly, right_poly


def highlight_lane_pixels(img, bird_eye):
    left_lane_x, left_lane_y, right_lane_x, right_lane_y = get_lane_pixels(bird_eye)

    result = np.zeros_like(bird_eye)

    for i, j in zip(left_lane_x, left_lane_y):
        result[j][i] = 255

    for i, j in zip(right_lane_x, right_lane_y):
        result[j][i] = 255

    dst_points = np.array([[0, 160], [320, 160], [114, 43], [172, 39]], np.float32)
    src_points = np.array([[200, 160], [400, 160], [140, 30], [360, 30]], np.float32)
    lambda_mat = cv2.getPerspectiveTransform(src_points, dst_points)

    result = cv2.warpPerspective(result, lambda_mat, (320, 160))
    # result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    tmp = img
    # print(tmp.shape)
    # print(result.shape)
    # print(tmp[:, :, 2].shape)
    tmp[:, :, 0] = cv2.bitwise_or(tmp[:, :, 0], result)
    # tmp[:, :, 1] = cv2.bitwise_or(tmp[:, :, 1], result)

    return tmp


def highlight_lane(img, bird_eye, left_poly, right_poly, idx):
    background = np.zeros((bird_eye.shape[0], bird_eye.shape[1], 3), dtype=np.uint8)

    ys = np.arange(bird_eye.shape[0] // 3, bird_eye.shape[0])
    for y in ys:
        left = fit_polynomial(y, left_poly)
        right = fit_polynomial(y, right_poly)
        # background[y, int(left):int(right)] = (0, 255, 0)

    dst_points = np.array([[0, 160], [320, 160], [114, 39], [172, 39]], np.float32)
    src_points = np.array([[200, 160], [400, 160], [140, 30], [360, 30]], np.float32)
    lambda_mat = cv2.getPerspectiveTransform(src_points, dst_points)

    background = cv2.warpPerspective(background, lambda_mat, (320, 160))

    target_points = []
    ys = ys[::len(ys) // 16]
    ys = ys[(idx % 2)::2]

    for y in ys:
        target_points.append([fit_polynomial(y, left_poly), y])
        target_points.append([fit_polynomial(y, right_poly), y])

    target_points = np.array([target_points])

    target_points = cv2.perspectiveTransform(target_points, lambda_mat)
    target_points = np.array(target_points[0], dtype=np.int32)
    for left_corner, right_corner in zip(target_points[::2], target_points[1::2]):
        tbot = left_corner[1]
        tleft = left_corner[0]
        tright = right_corner[0]
        twidth = tright - tleft
        theight = twidth // 2
        ttop = max(0, tbot - theight)
        cv2.line(background, (tleft, tbot), (tright, tbot), (0, 255, 0))
        cv2.line(background, (tleft, tbot), (tleft, ttop), (0, 255, 0))
        cv2.line(background, (tright, tbot), (tright, ttop), (0, 255, 0))
        cv2.line(background, (tleft, ttop), (tright, ttop), (0, 255, 0))

    return background


def lane_extraction(img, current_lanes, idx):
    bird_eye = bird_eye_transform(img)
    left_poly, right_poly = get_lane_polynomials(bird_eye)
    if current_lanes is not None:
        if left_poly is not None:
            left_poly = current_lanes[0] * config.decay_factor + left_poly * (1 - config.decay_factor)
            right_poly = current_lanes[1] * config.decay_factor + right_poly * (1 - config.decay_factor)
        else:
            [left_poly, right_poly] = current_lanes

    result = None
    if left_poly is not None:
        background = highlight_lane(img, bird_eye, left_poly, right_poly, idx)
        # return background
        result = cv2.addWeighted(img, 1, background, .5, 0)
        background = highlight_lane_pixels(img, bird_eye)
        result = cv2.addWeighted(result, 1, background, 1, 0)

    return result, [left_poly, right_poly]


# img = cv2.imread('../data/failed_img.jpg')
#
# bird_eye = bird_eye_transform(img)
# left_poly, right_poly = get_lane_polynomials(bird_eye)
# res = lane_extraction(img)
#
# plt.subplot(221)
# plt.imshow(img)
# plt.subplot(222)
# plt.imshow(bird_eye, cmap='gray')
# plt.subplot(223)
# plt.imshow(highlight_lane_pixels(bird_eye), cmap='gray')
# plt.subplot(224)
# plt.imshow(res)
# plt.show()
