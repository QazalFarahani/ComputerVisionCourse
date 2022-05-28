import cv2
import numpy as np


def structure_tensor(img, s):
    d_x, d_y = get_gradients(img, 3)
    d2_x, d2_y = d_x ** 2, d_y ** 2
    d_xy = d_x * d_y
    s2_x = cv2.GaussianBlur(d2_x, (21, 21), s)
    s2_y = cv2.GaussianBlur(d2_y, (21, 21), s)
    s_xy = cv2.GaussianBlur(d_xy, (21, 21), s)
    magnitude = cv2.magnitude(d_x, d_y)
    return s2_x, s_xy, s2_y, magnitude


def get_harris_score(src, s, k):
    s2_x, s_xy, s2_y, mag = structure_tensor(src, s)
    det = s2_x * s2_y - s_xy ** 2
    tr = s2_x + s2_y
    return det - k * tr ** 2, mag


def get_gradients(src, k_size):
    d_x = cv2.Sobel(src, cv2.CV_64F, 1, 0, ksize=k_size)
    d_y = cv2.Sobel(src, cv2.CV_64F, 0, 1, ksize=k_size)
    return d_x, d_y


def nms(src, block_size, threshold):
    coordinates = []
    while src.max() > threshold:
        coord = np.argmax(src)
        coord = np.unravel_index(coord, src.shape)
        coordinates.append(coord)
        src[max(0, coord[0] - block_size):min(src.shape[0], coord[0] + block_size),
            max(0, coord[1] - block_size):min(src.shape[1], coord[1] + block_size)] = 0
    return np.array(coordinates)[:, :2]


def draw_points(img, points):
    for p in points:
        cv2.circle(img, tuple(p[::-1]), color=(0, 0.4, 1), radius=10, thickness=-1)
    return img


def get_feature_vector(img, d, points):
    feature_vectors = []
    for p in points:
        fv = img[max(0, p[0] - d):min(img.shape[0], p[0] + d), max(0, p[1] - d):min(img.shape[1], p[1] + d)]
        feature_vectors.append(fv)
    return feature_vectors


def matched_points(distance, threshold):
    mp_1 = -np.ones(distance.shape[0], dtype=int)
    mp_2 = -np.ones(distance.shape[1], dtype=int)
    multiple = False
    for p in range(distance.shape[0]):
        sorted_d = np.argsort(distance[p, :])
        p_1, p_2 = sorted_d[0], sorted_d[1]
        d1, d2 = distance[p, [p_1, p_2]]
        if d1 / d2 < threshold:
            mp_1[p] = p_1
            if multiple:
                mp_1 = -1

    for p in range(distance.shape[1]):
        sorted_d = np.argsort(distance[:, p])
        p_1, p_2 = sorted_d[0], sorted_d[1]
        d1, d2 = distance[[p_1, p_2], p]
        if d1 / d2 < threshold:
            mp_2[p] = p_1
            if multiple:
                mp_2 = -1
    return mp_1, mp_2


def draw_lines(mp_1, mp_2, p_1, p_2, img_1, img_2):
    for i, m in enumerate(mp_1):
        if m != -1 and mp_2[m] == i:
            cv2.circle(img_1, tuple(p_1[i][::-1]), color=(0, 1, 0.8), radius=10, thickness=-1)
            cv2.circle(img_2, tuple(p_2[m][::-1]), color=(0, 1, 0.8), radius=10, thickness=-1)
    result = np.concatenate([img_1.copy(), img_2.copy()], axis=1)
    for i, m in enumerate(mp_1):
        if m != -1 and mp_2[m] == i:
            p1, p2 = tuple(p_1[i][::-1]), tuple(p_2[m][::-1] + np.array([img_1.shape[1], 0]))
            cv2.line(result, p1, p2, color=(0.5, 1, 0), thickness=2)
    return img_1, img_2, result


def process(img1, img2, sigma, k, threshold, block_size, d_vec, match_threshold):
    harris_scores_1, magnitude_1 = get_harris_score(img1, sigma, k)
    harris_scores_2, magnitude_2 = get_harris_score(img2, sigma, k)
    cv2.imwrite('res01_grad.jpg', cv2.normalize(magnitude_1, 0, 0, 255, cv2.NORM_MINMAX))
    cv2.imwrite('res02_grad.jpg', cv2.normalize(magnitude_2, 0, 0, 255, cv2.NORM_MINMAX))
    cv2.imwrite('res03_score.jpg', cv2.normalize(harris_scores_1, 0, 0, 255, cv2.NORM_MINMAX))
    cv2.imwrite('res04_score.jpg', cv2.normalize(harris_scores_2, 0, 0, 255, cv2.NORM_MINMAX))

    harris_threshold_1 = np.where(harris_scores_1 > threshold, harris_scores_1, 0)
    harris_threshold_2 = np.where(harris_scores_2 > threshold, harris_scores_2, 0)
    cv2.imwrite('res05_thresh.jpg', cv2.normalize(harris_threshold_1, 0, 0, 255, cv2.NORM_MINMAX))
    cv2.imwrite('res06_thresh.jpg', cv2.normalize(harris_threshold_2, 0, 0, 255, cv2.NORM_MINMAX))

    points_1 = nms(harris_threshold_1, block_size, threshold)
    points_2 = nms(harris_threshold_2, block_size, threshold)
    interest_points_on_1 = draw_points(img1.copy(), points_1)
    interest_points_on_2 = draw_points(img2.copy(), points_2)
    cv2.imwrite('res07_harris.jpg', cv2.normalize(interest_points_on_1, 0, 0, 255, cv2.NORM_MINMAX))
    cv2.imwrite('res08_harris.jpg', cv2.normalize(interest_points_on_2, 0, 0, 255, cv2.NORM_MINMAX))

    feature_vectors_1 = get_feature_vector(img1, d_vec, points_1)
    feature_vectors_2 = get_feature_vector(img2, d_vec, points_2)
    distances = np.array(
        [[np.linalg.norm(feature_vector1 - feature_vector2) for feature_vector2 in feature_vectors_2] for
         feature_vector1 in feature_vectors_1])

    matched_points_1, matched_points_2 = matched_points(distances, match_threshold)
    img1, img2, img_side_by_side = draw_lines(matched_points_1, matched_points_2, points_1, points_2, img1, img2)
    cv2.imwrite('res09_corres.jpg', cv2.normalize(img1, 0, 0, 255, cv2.NORM_MINMAX))
    cv2.imwrite('res10_corres.jpg', cv2.normalize(img2, 0, 0, 255, cv2.NORM_MINMAX))
    cv2.imwrite('res11.jpg', cv2.normalize(img_side_by_side, 0, 0, 255, cv2.NORM_MINMAX))


image_1 = cv2.imread('im01.jpg').astype(float)
image_2 = cv2.imread('im02.jpg').astype(float)
image_1 = image_1 / np.max(image_1)
image_2 = image_2 / np.max(image_2)

process(image_1, image_2, 5, 0.05, 0.01, 50, 30, 0.9)
